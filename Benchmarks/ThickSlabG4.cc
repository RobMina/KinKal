#include "CLHEP/Random/Random.h"
#include "G4Box.hh"
#include "G4EmCalculator.hh"
#include "G4EmParameters.hh"
#include "G4HadronicParameters.hh"
#include "G4IonisParamMat.hh"
#include "G4LogicalVolume.hh"
#include "G4NistManager.hh"
#include "G4PVPlacement.hh"
#include "G4ParticleGun.hh"
#include "G4ParticleTable.hh"
#include "G4PhysListFactory.hh"
#include "G4ProcessManager.hh"
#include "G4RunManager.hh"
#include "G4Step.hh"
#include "G4StepLimiter.hh"
#include "G4SystemOfUnits.hh"
#include "G4Track.hh"
#include "G4UserSteppingAction.hh"
#include "G4UserLimits.hh"
#include "G4VModularPhysicsList.hh"
#include "G4VPhysicalVolume.hh"
#include "G4VUserDetectorConstruction.hh"
#include "G4VUserPrimaryGeneratorAction.hh"

#include "KinKal/General/Parameters.hh"
#include "KinKal/Detector/ElementXing.hh"
#include "KinKal/General/MomBasis.hh"
#include "KinKal/MatEnv/DetMaterial.hh"
#include "KinKal/MatEnv/ElmPropObj.hh"
#include "KinKal/MatEnv/MatMaterialObj.hh"
#include "KinKal/MatEnv/MtrPropObj.hh"

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Config {
  int events = 20000;
  double momentum = 1000.0;   // MeV/c
  double thickness = 500.0;   // mm
  double maxStep = -1.0;       // mm; negative means no slab user step limit
  std::string material = "CONCRETE_MARS";
  std::string physicsList = "ShieldingM";
  long seed = 12345;
  bool predictOnly = false;    // skip Geant4 entirely; emit only the KinKal + two-plane predictions
  bool ionDiag = false;        // dump the KinKal-vs-Geant4 ionisation parameters (I, density effect)
};

constexpr double kMuonMassMeV = 105.6583745;

struct Moments {
  int n = 0;
  int stopped = 0;
  double sx = 0.0;
  double sy = 0.0;
  double stx = 0.0;
  double sty = 0.0;
  double sde = 0.0;
  double sx2 = 0.0;
  double sy2 = 0.0;
  double stx2 = 0.0;
  double sty2 = 0.0;
  double sde2 = 0.0;
  double sx_tx = 0.0;
  double sy_ty = 0.0;

  void add(double x, double y, double tx, double ty, double de) {
    ++n;
    sx += x; sy += y; stx += tx; sty += ty; sde += de;
    sx2 += x*x; sy2 += y*y; stx2 += tx*tx; sty2 += ty*ty; sde2 += de*de;
    sx_tx += x*tx; sy_ty += y*ty;
  }

  double mean(double sum) const { return sum/static_cast<double>(n); }
  double var(double sum, double sum2) const {
    double const mu = mean(sum);
    return sum2/static_cast<double>(n) - mu*mu;
  }
  double cov(double suma, double sumb, double sumab) const {
    return sumab/static_cast<double>(n) - mean(suma)*mean(sumb);
  }
};

struct ElementSpec {
  char const* name;
  double z;
  double a;
  double fraction; // mass fraction
};

struct MaterialSpec {
  double density; // g/cm3
  std::vector<ElementSpec> elements;
};

// Shared material registry used by BOTH the Geant4 detector and the KinKal DetMaterial, so the
// two sides see identical composition and density.  CONCRETE_MARS is the Mu2e ExtShield concrete;
// ALUMINUM and IRON span the low-Z / high-Z ends needed to exercise the dE/dx-vs-scattering
// Z-moment decoupling (design note section 1.1) — a concrete-only scan cannot test that.
MaterialSpec materialSpec(std::string const& name) {
  if(name == "CONCRETE_MARS") {
    return {2.35, {
      {"H",  1.0,  1.00794,  0.006}, {"C",  6.0, 12.0107,   0.030},
      {"O",  8.0, 15.9994,   0.500}, {"Na",11.0, 22.989770, 0.010},
      {"Al",13.0, 26.981538, 0.030}, {"Si",14.0, 28.0855,   0.200},
      {"K", 19.0, 39.0983,   0.010}, {"Ca",20.0, 40.078,    0.200},
      {"Fe",26.0, 55.845,    0.014}
    }};
  }
  if(name == "ALUMINUM") return {2.699, {{"Al",13.0,26.981538,1.0}}};
  if(name == "IRON")     return {7.874, {{"Fe",26.0,55.845,  1.0}}};
  // --- Mu2eKinKal composite materials along the tracker->CRV path, from
  //     Offline/TrackerConditions/data/MaterialsList.data, resolved to element mass fractions. ---
  // DS coils (collapsed), Al + NbTi(0.65 Nb / 0.35 Ti).  Wide Z-spread (13,41,22): predicted +8% scatter.
  if(name == "DSCOILMIX") return {3.20323, {
    {"Al",13.0,26.981539,0.732270}, {"Nb",41.0,92.906380,0.174025}, {"Ti",22.0,47.880000,0.093706}}};
  // DS cryostat shells.  Near-mono-Z (Cr/Mn/Fe/Ni 24-28 + 1% Si): predicted ~0% (the control case).
  if(name == "STAINLESSSTEEL") return {8.02, {
    {"Mn",25.0,54.938049,0.02}, {"Si",14.0,28.085500,0.01}, {"Cr",24.0,51.996100,0.19},
    {"Ni",28.0,58.693400,0.10}, {"Fe",26.0,55.845000,0.68}}};
  // Tracker straw wall (Mylar + Au + Al traces).  Low-Z C/O/H with a Z=79 trace: predicted +18% scatter.
  if(name == "STRAWWALL") return {1.4325, {
    {"C", 6.0,12.011000,0.605961}, {"O", 8.0,15.999400,0.322870}, {"H", 1.0,1.007940,0.040680},
    {"Au",79.0,196.966540,0.018}, {"Al",13.0,26.981539,0.012489}}};
  throw std::runtime_error("Unknown material " + name);
}

std::unique_ptr<MatEnv::DetMaterial> makeKinKalMaterial(std::string const& name) {
  MaterialSpec const spec = materialSpec(name);
  auto const& els = spec.elements;

  std::vector<int> iflg(els.size(), 0);
  std::vector<double> weights;
  std::vector<std::string> componentNames;
  weights.reserve(els.size());
  componentNames.reserve(els.size());
  for(auto const& elem : els) {
    weights.push_back(elem.fraction);
    componentNames.emplace_back(elem.name);
  }

  MatEnv::MatMaterialObj material(name, spec.density, 0.0, 0.0,
      static_cast<int>(els.size()), &iflg, &weights,
      &componentNames, -1.0, -1.0, 0.0, 0.0, 0.0, "solid");
  auto properties = std::make_unique<MatEnv::MtrPropObj>(&material);

  std::vector<std::unique_ptr<MatEnv::ElmPropObj>> elements;
  elements.reserve(els.size());
  for(auto const& elem : els) {
    elements.emplace_back(std::make_unique<MatEnv::ElmPropObj>(elem.name, elem.z, elem.a));
    properties->AddElement(elements.back().get(), elem.fraction);
  }

  MatEnv::DetMaterialConfig config;
  config.elossmode_ = MatEnv::DetMaterial::moyalmean;
  return std::make_unique<MatEnv::DetMaterial>(name.c_str(), properties.get(), config);
}

// Build the Geant4 material from the shared registry (no RunManager needed; G4IonisParamMat — the
// mean excitation energy and density-effect parameters — is computed at G4Material construction).
G4Material* buildSlabMaterial(std::string const& name) {
  if(auto* existing = G4Material::GetMaterial(name, false)) return existing;
  MaterialSpec const spec = materialSpec(name);
  auto* nist = G4NistManager::Instance();
  auto* mat = new G4Material(name, spec.density*CLHEP::g/CLHEP::cm3,
                             static_cast<int>(spec.elements.size()));
  for(auto const& elem : spec.elements)
    mat->AddElement(nist->FindOrBuildElement(elem.name), elem.fraction);
  return mat;
}

class StraightTrajectory {
public:
  StraightTrajectory(double momentum, double mass) : momentum_(momentum), mass_(mass) {}
  double momentum() const { return momentum_; }
  double mass() const { return mass_; }
  double energy() const { return std::sqrt(momentum_*momentum_ + mass_*mass_); }
  KinKal::VEC3 direction(double) const { return KinKal::VEC3(0.0, 0.0, 1.0); }
  KinKal::VEC3 direction(double, KinKal::MomBasis::Direction dir) const {
    if(dir == KinKal::MomBasis::momdir_) return KinKal::VEC3(0.0, 0.0, 1.0);
    if(dir == KinKal::MomBasis::perpdir_) return KinKal::VEC3(1.0, 0.0, 0.0);
    return KinKal::VEC3(0.0, 1.0, 0.0);
  }

private:
  double momentum_;
  double mass_;
};

class BenchmarkXing : public KinKal::ElementXing<StraightTrajectory> {
public:
  BenchmarkXing(MatEnv::DetMaterial const& material, double pathlen,
      double momentum, double mass) :
    traj_(momentum, mass), xings_{KinKal::MaterialXing(material, pathlen)} {}

  void updateReference(PTRAJ const&) override {}
  void updateState(KinKal::MetaIterConfig const&, bool) override {}
  KinKal::Parameters params() const override { return KinKal::Parameters(); }
  double time() const override { return 0.0; }
  double transitTime() const override { return 0.0; }
  StraightTrajectory const& referenceTrajectory() const override { return traj_; }
  std::vector<KinKal::MaterialXing> const& matXings() const override { return xings_; }
  void print(std::ostream& ost=std::cout, int=0) const override { ost << "BenchmarkXing"; }
  bool active() const override { return true; }
  bool thickScatter() const override { return true; }

private:
  StraightTrajectory traj_;
  std::vector<KinKal::MaterialXing> xings_;
};

struct KinKalPrediction {
  double radlen = 0.0;
  double xOverX0 = 0.0;
  double energyLoss = 0.0;
  double energyLossRms = 0.0;
  double scatterAngleVar = 0.0;
  double sigmaTheta = 0.0;
  double sigmaPos = 0.0;
  double posMomCov = 0.0;
  double corr = 0.0;
  double dmomZ = 0.0;
  double covXX = 0.0;
  double covPX = 0.0;
  double covPXPX = 0.0;
  double kappa = 0.0;
  double peff = 0.0;
  unsigned nstep = 0;
  double peffSingle = 0.0;
  double scatSingle = 0.0;
  double dESingle = 0.0;
};

KinKalPrediction runKinKalPrediction(Config const& cfg, double muonMass) {
  auto material = makeKinKalMaterial(cfg.material);
  BenchmarkXing xing(*material, cfg.thickness, cfg.momentum, muonMass);
  KinKal::SVEC3 dmom;
  KinKal::DMAT cov;
  xing.particleStateChange(dmom, cov, true);
  // pull the momentum-integrated scalars from the same code path the covariance is built from
  KinKal::ThickXingEffects const eff = xing.thickEffects();

  KinKalPrediction pred;
  pred.radlen = material->radiationLength()/material->density();
  pred.xOverX0 = material->radiationFraction(cfg.thickness);
  // mean / RMS energy loss now come from the sub-stepped Bethe-Bloch mean + Gaussian straggling
  pred.energyLoss = -eff.dE;
  pred.energyLossRms = std::sqrt(std::max(0.0, eff.dEVar));
  // scattering width is the momentum-integrated (effective-momentum, full-path) projected variance
  pred.scatterAngleVar = eff.scatvar;
  pred.sigmaTheta = std::sqrt(std::max(0.0, eff.scatvar));
  pred.sigmaPos = std::sqrt(cov(0,0));
  pred.posMomCov = cov(0,3);
  pred.corr = cov(0,3)/(std::sqrt(cov(0,0))*std::sqrt(cov(3,3)));
  pred.dmomZ = dmom[2];
  pred.covXX = cov(0,0);
  pred.covPX = cov(0,3);
  pred.covPXPX = cov(3,3);
  // diagnostics from the momentum integration
  pred.kappa = eff.kappa;
  pred.peff = eff.peff;
  pred.nstep = eff.nstep;
  pred.peffSingle = cfg.momentum;
  pred.scatSingle = std::sqrt(material->scatterAngleVar(cfg.momentum, cfg.thickness, muonMass));
  pred.dESingle = -material->energyLoss(cfg.momentum, cfg.thickness, muonMass);
  return pred;
}

// Faithful reproduction of the Mu2e KinKalGeomMaker "two-plane" workaround (design note section 3):
// the slab is split into two thin Gauss-point scatterers at center +/- t/(2 sqrt3), each carrying
// HALF the material, propagated to the exit by the extrapolator's transport.  This is the prior
// state of the art in Mu2eKinKal.  It uses the UNMODIFIED thin physics (Moyal mean energy loss,
// Lynch-Dahl scattering, Moyal straggling), evaluated over half-thickness at each plane, with the
// momentum tracked from plane 1 to plane 2 (a coarse 2-point integration).
struct TwoPlanePrediction {
  double sigmaTheta = 0.0;
  double sigmaPos = 0.0;
  double corr = 0.0;
  double energyLoss = 0.0;
  double energyLossRms = 0.0;
};

TwoPlanePrediction runTwoPlanePrediction(Config const& cfg, double muonMass) {
  auto material = makeKinKalMaterial(cfg.material);
  double const L = cfg.thickness;
  double const half = 0.5*L;                 // material path carried by each plane
  double const offset = L/(2.0*std::sqrt(3.0));
  double const s1 = 0.5*L - offset;          // Gauss-point positions from the slab entry
  double const s2 = 0.5*L + offset;
  double const lever1 = L - s1;              // distance from each plane to the exit face
  double const lever2 = L - s2;

  double const p0 = cfg.momentum;
  double const E0 = std::sqrt(p0*p0 + muonMass*muonMass);
  // plane 1 at the entrance momentum
  double const sv1 = material->scatterAngleVar(p0, half, muonMass);
  double const de1 = material->energyLoss(p0, half, muonMass);     // <0 (Moyal mean)
  double const dev1 = material->energyLossVar(p0, half, muonMass);
  // advance the momentum through plane 1's mean loss, then plane 2
  double const E1 = E0 + de1;
  double const p1 = std::sqrt(std::max(0.0, E1*E1 - muonMass*muonMass));
  double const sv2 = material->scatterAngleVar(p1, half, muonMass);
  double const de2 = material->energyLoss(p1, half, muonMass);
  double const dev2 = material->energyLossVar(p1, half, muonMass);

  double const scatvar = sv1 + sv2;                                  // exit angle variance
  double const perpposvar = lever1*lever1*sv1 + lever2*lever2*sv2;   // transported displacement
  double const perpmomvar = p0*p0*sv1 + p1*p1*sv2;                   // transverse momentum
  double const perpposmomcov = lever1*p0*sv1 + lever2*p1*sv2;

  TwoPlanePrediction tp;
  tp.sigmaTheta = std::sqrt(std::max(0.0, scatvar));
  tp.sigmaPos = std::sqrt(std::max(0.0, perpposvar));
  tp.corr = perpposmomcov/std::sqrt(perpposvar*perpmomvar);
  tp.energyLoss = -(de1 + de2);
  tp.energyLossRms = std::sqrt(std::max(0.0, dev1 + dev2));
  return tp;
}

class SlabDetector : public G4VUserDetectorConstruction {
public:
  explicit SlabDetector(Config const& cfg) : cfg_(cfg) {}

  G4VPhysicalVolume* Construct() override {
    auto* nist = G4NistManager::Instance();
    auto* vacuum = nist->FindOrBuildMaterial("G4_Galactic");
    slabMaterial_ = findOrBuildMaterial(cfg_.material);
    if(slabMaterial_ == nullptr) throw std::runtime_error("Unknown material " + cfg_.material);

    double const worldXY = 6000.0*mm;
    double const worldZ = (cfg_.thickness + 2000.0)*mm;
    auto* worldSolid = new G4Box("WorldSolid", 0.5*worldXY, 0.5*worldXY, 0.5*worldZ);
    auto* worldLV = new G4LogicalVolume(worldSolid, vacuum, "World");
    auto* worldPV = new G4PVPlacement(nullptr, {}, worldLV, "World", nullptr, false, 0, true);

    auto* slabSolid = new G4Box("SlabSolid", 0.25*worldXY, 0.25*worldXY,
                                0.5*cfg_.thickness*mm);
    auto* slabLV = new G4LogicalVolume(slabSolid, slabMaterial_, "Slab");
    if(cfg_.maxStep > 0.0){
      slabLV->SetUserLimits(new G4UserLimits(cfg_.maxStep*mm));
    }
    new G4PVPlacement(nullptr, {}, slabLV, "Slab", worldLV, false, 0, true);
    return worldPV;
  }

  G4Material const* slabMaterial() const { return slabMaterial_; }

private:
  static G4Material* findOrBuildMaterial(std::string const& name) {
    return buildSlabMaterial(name); // shared registry: Geant4 and KinKal use identical composition
  }

  Config const& cfg_;
  G4Material* slabMaterial_ = nullptr;
};

class PrimaryGun : public G4VUserPrimaryGeneratorAction {
public:
  explicit PrimaryGun(Config const& cfg) : cfg_(cfg), gun_(1) {
    auto* particle = G4ParticleTable::GetParticleTable()->FindParticle("mu-");
    gun_.SetParticleDefinition(particle);
    gun_.SetParticleMomentum(cfg_.momentum*MeV);
    gun_.SetParticleMomentumDirection({0.0, 0.0, 1.0});
    gun_.SetParticlePosition({0.0, 0.0, (-0.5*cfg_.thickness - 1.0)*mm});
  }

  void GeneratePrimaries(G4Event* event) override {
    gun_.GeneratePrimaryVertex(event);
  }

private:
  Config const& cfg_;
  G4ParticleGun gun_;
};

class ExitRecorder : public G4UserSteppingAction {
public:
  ExitRecorder(Config const& cfg, Moments& moments) : cfg_(cfg), moments_(moments) {}

  void UserSteppingAction(G4Step const* step) override {
    auto* track = step->GetTrack();
    if(track->GetTrackID() != 1) return;

    auto const* pre = step->GetPreStepPoint();
    auto const* post = step->GetPostStepPoint();
    auto const* preVol = pre->GetPhysicalVolume();
    auto const* postVol = post->GetPhysicalVolume();
    bool const inSlab = preVol != nullptr && preVol->GetName() == "Slab";
    bool const leftSlab = inSlab && (postVol == nullptr || postVol->GetName() != "Slab");

    if(leftSlab){
      auto const pos = post->GetPosition();
      auto const mom = post->GetMomentum();
      double const mass = track->GetDefinition()->GetPDGMass()/MeV;
      double const e0 = std::sqrt(cfg_.momentum*cfg_.momentum + mass*mass);
      double const pExit = mom.mag()/MeV;
      double const eExit = std::sqrt(pExit*pExit + mass*mass);
      moments_.add(pos.x()/mm, pos.y()/mm, mom.x()/mom.z(), mom.y()/mom.z(), e0 - eExit);
      track->SetTrackStatus(fStopAndKill);
    } else if(inSlab && track->GetKineticEnergy() <= 0.0){
      ++moments_.stopped;
    }
  }

private:
  Config const& cfg_;
  Moments& moments_;
};

class Mu2eStepLimiterPhysics : public G4VPhysicsConstructor {
public:
  Mu2eStepLimiterPhysics() : G4VPhysicsConstructor("Mu2eG4StepLimiterPhysicsConstructor") {}

  void ConstructParticle() override {}

  void ConstructProcess() override {
    static char const* const names[] = {
      "e+", "e-", "mu+", "mu-", "pi+", "pi-", "kaon+", "kaon-",
      "proton", "anti_proton", "chargedgeantino", "deuteron", "anti_deuteron",
      "triton", "anti_triton", "He3", "anti_He3", "alpha", "anti_alpha"
    };
    auto* particles = G4ParticleTable::GetParticleTable();
    auto* iter = particles->GetIterator();
    iter->reset();
    while((*iter)()){
      auto* particle = iter->value();
      auto const particleName = particle->GetParticleName();
      for(auto const* name : names){
        if(particleName == name){
          particle->GetProcessManager()->AddDiscreteProcess(new G4StepLimiter);
          break;
        }
      }
    }
  }
};

G4VUserPhysicsList* makeMu2ePhysicsList(Config const& cfg) {
  G4HadronicParameters::Instance()->SetBertiniAs11_2(true);
  G4PhysListFactory factory;
  factory.SetVerbose(0);
  auto* physics = factory.GetReferencePhysList(cfg.physicsList);
  if(physics == nullptr) throw std::runtime_error("Unable to load physics list " + cfg.physicsList);

  physics->SetVerboseLevel(0);
  physics->RegisterPhysics(new Mu2eStepLimiterPhysics);

  auto* em = G4EmParameters::Instance();
  em->SetVerbose(0);
  em->SetMuHadLateralDisplacement(true);

  physics->SetDefaultCutValue(0.010*mm);
  physics->SetCutValue(0.0*mm, "proton");
  return physics;
}

Config parseArgs(int argc, char** argv) {
  Config cfg;
  for(int i=1; i<argc; ++i){
    std::string const arg = argv[i];
    auto value = [&](std::string const& name) -> std::string {
      if(i+1 >= argc) throw std::runtime_error("Missing value for " + name);
      return argv[++i];
    };
    if(arg == "--events") cfg.events = std::atoi(value(arg).c_str());
    else if(arg == "--momentum") cfg.momentum = std::atof(value(arg).c_str());
    else if(arg == "--thickness") cfg.thickness = std::atof(value(arg).c_str());
    else if(arg == "--max-step") cfg.maxStep = std::atof(value(arg).c_str());
    else if(arg == "--material") cfg.material = value(arg);
    else if(arg == "--physics-list") cfg.physicsList = value(arg);
    else if(arg == "--seed") cfg.seed = std::atol(value(arg).c_str());
    else if(arg == "--predict-only") cfg.predictOnly = true;
    else if(arg == "--ion-diag") cfg.ionDiag = true;
    else if(arg == "--help"){
      std::cout << "Usage: ThickSlabG4 [--events N] [--momentum MeV] "
                   "[--thickness mm] [--max-step mm] [--material NAME] "
                   "[--physics-list NAME] [--seed N] [--predict-only]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown argument " + arg);
    }
  }
  return cfg;
}

double highlandTheta0(double momentum, double mass, double thickness, double radlen) {
  double const beta = momentum/std::sqrt(momentum*momentum + mass*mass);
  double const xOverX0 = thickness/radlen;
  return (13.6/(beta*momentum))*std::sqrt(xOverX0)*(1.0 + 0.038*std::log(xOverX0/(beta*beta)));
}

} // namespace

int main(int argc, char** argv) {
  Config const cfg = parseArgs(argc, argv);

  if(cfg.predictOnly){
    // Geant4-free path: the KinKal + two-plane predictions are deterministic, so the full
    // material x thickness x momentum grid can be swept in milliseconds per cell.  Geant4 is
    // reserved for a validation subset (run without --predict-only).
    auto const kinkal = runKinKalPrediction(cfg, kMuonMassMeV);
    auto const twoplane = runTwoPlanePrediction(cfg, kMuonMassMeV);
    std::cout << std::setprecision(6)
              << "predict_only 1\n"
              << "material " << cfg.material << "\n"
              << "momentum_MeVc " << cfg.momentum << "\n"
              << "thickness_mm " << cfg.thickness << "\n"
              << "kinkal_radlen_mm " << kinkal.radlen << "\n"
              << "kinkal_x_over_x0 " << kinkal.xOverX0 << "\n"
              << "kinkal_nsubsteps " << kinkal.nstep << "\n"
              << "kinkal_vavilov_kappa " << kinkal.kappa << "\n"
              << "kinkal_p_eff_MeVc " << kinkal.peff << "\n"
              << "kinkal_sigma_theta " << kinkal.sigmaTheta << "\n"
              << "kinkal_sigma_theta_singlestep " << kinkal.scatSingle << "\n"
              << "kinkal_sigma_pos_mm " << kinkal.sigmaPos << "\n"
              << "kinkal_corr_x_theta " << kinkal.corr << "\n"
              << "kinkal_mean_dE_MeV " << kinkal.energyLoss << "\n"
              << "kinkal_mean_dE_singlestep_MeV " << kinkal.dESingle << "\n"
              << "kinkal_rms_dE_MeV " << kinkal.energyLossRms << "\n"
              << "twoplane_sigma_theta " << twoplane.sigmaTheta << "\n"
              << "twoplane_sigma_pos_mm " << twoplane.sigmaPos << "\n"
              << "twoplane_corr_x_theta " << twoplane.corr << "\n"
              << "twoplane_mean_dE_MeV " << twoplane.energyLoss << "\n"
              << "twoplane_rms_dE_MeV " << twoplane.energyLossRms << "\n";
    return 0;
  }

  CLHEP::HepRandom::setTheSeed(cfg.seed);

  auto* runManager = new G4RunManager;
  auto* detector = new SlabDetector(cfg);
  Moments moments;
  runManager->SetUserInitialization(detector);
  runManager->SetUserInitialization(makeMu2ePhysicsList(cfg));
  runManager->SetUserAction(new PrimaryGun(cfg));
  runManager->SetUserAction(new ExitRecorder(cfg, moments));
  runManager->Initialize();

  if(cfg.ionDiag){
    // Pin the ~2% dE offset: dump KinKal's ionisation inputs (mean excitation energy I and the
    // Sternheimer density-effect coefficients + delta(betagamma)) and compare the resulting
    // ionization dE/dx to Geant4's, point-by-point in betagamma, via G4EmCalculator (the EM physics
    // tables are built by the BeamOn(1) below).  A flat dE/dx residual fingerprints an I offset; a
    // betagamma-dependent residual fingerprints the density-effect (delta).
    runManager->BeamOn(1); // build the EM physics tables for G4EmCalculator
    auto kk = makeKinKalMaterial(cfg.material);
    auto const* muon = G4ParticleTable::GetParticleTable()->FindParticle("mu-");
    double const mass = muon->GetPDGMass()/MeV;
    auto const* g4mat = detector->slabMaterial();
    auto* ion = g4mat->GetIonisation();
    G4EmCalculator emcalc;
    std::cout << std::setprecision(6)
              << "ion_diag 1\n"
              << "material " << cfg.material << "\n"
              << "I_kinkal_eV " << kk->eexc()*1.0e6 << "\n"
              << "I_g4_eV " << ion->GetMeanExcitationEnergy()/CLHEP::eV << "\n"
              << "# Sternheimer density-effect coefficients: x0 x1 Cbar a m\n"
              << "sternheimer_kinkal " << kk->X0() << " " << kk->X1() << " "
                                       << kk->bigC() << " " << kk->aFactor() << " " << kk->mPower() << "\n"
              << "sternheimer_g4 " << ion->GetX0density() << " " << ion->GetX1density() << " "
                                   << ion->GetCdensity() << " " << ion->GetAdensity() << " "
                                   << ion->GetMdensity() << "\n"
              << "# p(MeV/c) betagamma delta_kk delta_g4 dEdx_kk_MeVmm dEdx_g4_MeVmm resid\n";
    for(double p : {300.0, 600.0, 1000.0, 2000.0, 4000.0, 8000.0}){
      double const bg = p/mass;
      double const bg2 = bg*bg;
      double const x = std::log10(bg);
      double const dkk = kk->densityCorrection(bg2);
      double const dg4 = ion->DensityCorrection(x);
      double const dedx_kk = -kk->ionizationEnergyLossMean(p, 1.0, mass); // MeV/mm
      double const kinE = std::sqrt(p*p + mass*mass) - mass;
      double const dedx_g4 = emcalc.ComputeTotalDEDX(kinE*MeV, muon, g4mat)/(MeV/mm);
      double const resid = (dedx_kk - dedx_g4)/dedx_g4;
      std::cout << "iondedx " << p << " " << bg << " " << dkk << " " << dg4 << " "
                << dedx_kk << " " << dedx_g4 << " " << resid << "\n";
    }
    delete runManager;
    return 0;
  }

  runManager->BeamOn(cfg.events);

  if(moments.n == 0) throw std::runtime_error("No primary muons exited the slab");

  auto const* muon = G4ParticleTable::GetParticleTable()->FindParticle("mu-");
  double const mass = muon->GetPDGMass()/MeV;
  double const kinE = std::sqrt(cfg.momentum*cfg.momentum + mass*mass) - mass;
  double const radlen = detector->slabMaterial()->GetRadlen()/mm;
  double const theta0 = highlandTheta0(cfg.momentum, mass, cfg.thickness, radlen);
  double const sigmaPos = cfg.thickness*theta0/std::sqrt(3.0);

  G4EmCalculator emcalc;
  double const dedx = emcalc.ComputeTotalDEDX(kinE*MeV, muon, detector->slabMaterial())/(MeV/mm);
  double const predDE = dedx*cfg.thickness;
  auto const kinkal = runKinKalPrediction(cfg, mass);
  auto const twoplane = runTwoPlanePrediction(cfg, mass);

  double const sx = std::sqrt(moments.var(moments.sx, moments.sx2));
  double const sy = std::sqrt(moments.var(moments.sy, moments.sy2));
  double const stx = std::sqrt(moments.var(moments.stx, moments.stx2));
  double const sty = std::sqrt(moments.var(moments.sty, moments.sty2));
  double const sde = std::sqrt(moments.var(moments.sde, moments.sde2));
  double const corrX = moments.cov(moments.sx, moments.stx, moments.sx_tx)/(sx*stx);
  double const corrY = moments.cov(moments.sy, moments.sty, moments.sy_ty)/(sy*sty);

  std::cout << std::setprecision(6)
            << "events_requested " << cfg.events << "\n"
            << "events_exited " << moments.n << "\n"
            << "material " << cfg.material << "\n"
            << "physics_list " << cfg.physicsList << "\n"
            << "mu2e_min_range_cut_mm 0.010\n"
            << "mu2e_proton_production_cut_mm 0\n"
            << "mu2e_mu_had_lateral_displacement 1\n"
            << "mu2e_bertini_as_11_2 1\n"
            << "momentum_MeVc " << cfg.momentum << "\n"
            << "thickness_mm " << cfg.thickness << "\n"
            << "max_step_mm " << cfg.maxStep << "\n"
            << "radlen_mm " << radlen << "\n"
            << "x_over_x0 " << cfg.thickness/radlen << "\n"
            << "theta0_pred " << theta0 << "\n"
            << "kinkal_radlen_mm " << kinkal.radlen << "\n"
            << "kinkal_x_over_x0 " << kinkal.xOverX0 << "\n"
            << "kinkal_nsubsteps " << kinkal.nstep << "\n"
            << "kinkal_vavilov_kappa " << kinkal.kappa << "\n"
            << "kinkal_p_eff_MeVc " << kinkal.peff << "\n"
            << "kinkal_scatter_angle_var " << kinkal.scatterAngleVar << "\n"
            << "kinkal_sigma_theta " << kinkal.sigmaTheta << "\n"
            << "kinkal_sigma_theta_singlestep " << kinkal.scatSingle << "\n"
            << "kinkal_mean_dE_singlestep_MeV " << kinkal.dESingle << "\n"
            << "twoplane_sigma_theta " << twoplane.sigmaTheta << "\n"
            << "twoplane_sigma_pos_mm " << twoplane.sigmaPos << "\n"
            << "twoplane_corr_x_theta " << twoplane.corr << "\n"
            << "twoplane_mean_dE_MeV " << twoplane.energyLoss << "\n"
            << "twoplane_rms_dE_MeV " << twoplane.energyLossRms << "\n"
            << "kinkal_sigma_pos_mm " << kinkal.sigmaPos << "\n"
            << "kinkal_corr_x_theta " << kinkal.corr << "\n"
            << "kinkal_mean_dE_MeV " << kinkal.energyLoss << "\n"
            << "kinkal_rms_dE_MeV " << kinkal.energyLossRms << "\n"
            << "kinkal_particle_state_dmom_z_MeVc " << kinkal.dmomZ << "\n"
            << "kinkal_cov_xx_mm2 " << kinkal.covXX << "\n"
            << "kinkal_cov_x_px_mm_MeVc " << kinkal.covPX << "\n"
            << "kinkal_cov_px_px_MeVc2 " << kinkal.covPXPX << "\n"
            << "sigma_theta_x_g4 " << stx << "\n"
            << "sigma_theta_y_g4 " << sty << "\n"
            << "sigma_pos_pred_mm " << sigmaPos << "\n"
            << "sigma_x_g4_mm " << sx << "\n"
            << "sigma_y_g4_mm " << sy << "\n"
            << "corr_pred " << std::sqrt(3.0)/2.0 << "\n"
            << "corr_x_theta_x_g4 " << corrX << "\n"
            << "corr_y_theta_y_g4 " << corrY << "\n"
            << "mean_dE_g4_MeV " << moments.mean(moments.sde) << "\n"
            << "rms_dE_g4_MeV " << sde << "\n"
            << "dedx_pred_MeV_per_mm " << dedx << "\n"
            << "mean_dE_pred_MeV " << predDE << "\n";

  delete runManager;
  return 0;
}
