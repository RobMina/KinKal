#include "CLHEP/Random/Random.h"
#include "G4Box.hh"
#include "G4EmCalculator.hh"
#include "G4EmParameters.hh"
#include "G4HadronicParameters.hh"
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
};

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
  double fraction;
};

std::vector<ElementSpec> concreteMarsElements() {
  return {
    {"H",  1.0,  1.00794,  0.006},
    {"C",  6.0, 12.0107,   0.030},
    {"O",  8.0, 15.9994,   0.500},
    {"Na",11.0, 22.989770, 0.010},
    {"Al",13.0, 26.981538, 0.030},
    {"Si",14.0, 28.0855,   0.200},
    {"K", 19.0, 39.0983,   0.010},
    {"Ca",20.0, 40.078,    0.200},
    {"Fe",26.0, 55.845,    0.014}
  };
}

std::unique_ptr<MatEnv::DetMaterial> makeKinKalMaterial(std::string const& name) {
  if(name != "CONCRETE_MARS") {
    throw std::runtime_error("KinKal side currently supports CONCRETE_MARS only");
  }

  std::vector<int> iflg(concreteMarsElements().size(), 0);
  std::vector<double> weights;
  std::vector<std::string> componentNames;
  weights.reserve(concreteMarsElements().size());
  componentNames.reserve(concreteMarsElements().size());
  for(auto const& elem : concreteMarsElements()) {
    weights.push_back(elem.fraction);
    componentNames.emplace_back(elem.name);
  }

  MatEnv::MatMaterialObj material(name, 2.35, 0.0, 0.0,
      static_cast<int>(concreteMarsElements().size()), &iflg, &weights,
      &componentNames, -1.0, -1.0, 0.0, 0.0, 0.0, "solid");
  auto properties = std::make_unique<MatEnv::MtrPropObj>(&material);

  std::vector<std::unique_ptr<MatEnv::ElmPropObj>> elements;
  elements.reserve(concreteMarsElements().size());
  for(auto const& elem : concreteMarsElements()) {
    elements.emplace_back(std::make_unique<MatEnv::ElmPropObj>(elem.name, elem.z, elem.a));
    properties->AddElement(elements.back().get(), elem.fraction);
  }

  MatEnv::DetMaterialConfig config;
  config.elossmode_ = MatEnv::DetMaterial::moyalmean;
  return std::make_unique<MatEnv::DetMaterial>(name.c_str(), properties.get(), config);
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
};

KinKalPrediction runKinKalPrediction(Config const& cfg, double muonMass) {
  auto material = makeKinKalMaterial(cfg.material);
  BenchmarkXing xing(*material, cfg.thickness, cfg.momentum, muonMass);
  KinKal::SVEC3 dmom;
  KinKal::DMAT cov;
  xing.particleStateChange(dmom, cov, true);

  KinKalPrediction pred;
  pred.radlen = material->radiationLength()/material->density();
  pred.xOverX0 = material->radiationFraction(cfg.thickness);
  pred.energyLoss = -material->energyLoss(cfg.momentum, cfg.thickness, muonMass);
  pred.energyLossRms = material->energyLossRMS(cfg.momentum, cfg.thickness, muonMass);
  pred.scatterAngleVar = material->scatterAngleVar(cfg.momentum, cfg.thickness, muonMass);
  pred.sigmaTheta = std::sqrt(pred.scatterAngleVar);
  pred.sigmaPos = std::sqrt(cov(0,0));
  pred.posMomCov = cov(0,3);
  pred.corr = cov(0,3)/(std::sqrt(cov(0,0))*std::sqrt(cov(3,3)));
  pred.dmomZ = dmom[2];
  pred.covXX = cov(0,0);
  pred.covPX = cov(0,3);
  pred.covPXPX = cov(3,3);
  return pred;
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
    if(name == "CONCRETE_MARS"){
      if(auto* existing = G4Material::GetMaterial(name, false)) return existing;
      auto* nist = G4NistManager::Instance();
      auto* concrete = new G4Material(name, 2.35*g/cm3, 9);
      concrete->AddElement(nist->FindOrBuildElement("H"),  0.006);
      concrete->AddElement(nist->FindOrBuildElement("C"),  0.030);
      concrete->AddElement(nist->FindOrBuildElement("O"),  0.500);
      concrete->AddElement(nist->FindOrBuildElement("Na"), 0.010);
      concrete->AddElement(nist->FindOrBuildElement("Al"), 0.030);
      concrete->AddElement(nist->FindOrBuildElement("Si"), 0.200);
      concrete->AddElement(nist->FindOrBuildElement("K"),  0.010);
      concrete->AddElement(nist->FindOrBuildElement("Ca"), 0.200);
      concrete->AddElement(nist->FindOrBuildElement("Fe"), 0.014);
      return concrete;
    }
    return G4NistManager::Instance()->FindOrBuildMaterial(name);
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
    else if(arg == "--help"){
      std::cout << "Usage: ThickSlabG4 [--events N] [--momentum MeV] "
                   "[--thickness mm] [--max-step mm] [--material NAME] "
                   "[--physics-list NAME] [--seed N]\n";
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
  CLHEP::HepRandom::setTheSeed(cfg.seed);

  auto* runManager = new G4RunManager;
  auto* detector = new SlabDetector(cfg);
  Moments moments;
  runManager->SetUserInitialization(detector);
  runManager->SetUserInitialization(makeMu2ePhysicsList(cfg));
  runManager->SetUserAction(new PrimaryGun(cfg));
  runManager->SetUserAction(new ExitRecorder(cfg, moments));
  runManager->Initialize();
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
            << "kinkal_scatter_angle_var " << kinkal.scatterAngleVar << "\n"
            << "kinkal_sigma_theta " << kinkal.sigmaTheta << "\n"
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
