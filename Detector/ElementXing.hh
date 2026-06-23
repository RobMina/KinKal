#ifndef KinKal_ElementXing_hh
#define KinKal_ElementXing_hh
//
//  Describe the material effects of a particle crossing a detector element (piece of the detector)
//  Used in the kinematic Kalman fit
//
#include "KinKal/General/CloneContext.hh"
#include "KinKal/General/MomBasis.hh"
#include "KinKal/Detector/MaterialXing.hh"
#include "KinKal/Trajectory/ParticleTrajectory.hh"
#include "KinKal/Fit/MetaIterConfig.hh"
#include <cmath>
#include <vector>
#include <array>
#include <algorithm>
#include <ostream>
#include <stdexcept>
#include <string>

namespace KinKal {
  // Accumulated material effects of a (thick) element crossing, integrated along the path with
  // the particle momentum tracked through the energy loss.  The "perp" quantities are per
  // transverse projection; the displacement terms (perpposvar, perpposmomcov) carry the PDG
  // correlated multiple-scattering covariance.  All energy/momentum changes are signed
  // (negative == loss) to match DetMaterial's convention.
  struct ThickXingEffects {
    double dmom = 0.0;          // total momentum change along the momentum direction (<0 for loss)
    double dE = 0.0;            // total MEAN energy loss (<0)
    double dEVar = 0.0;         // energy-loss variance
    double paramomvar = 0.0;    // longitudinal momentum variance (energy straggling)
    double scatvar = 0.0;       // total projected multiple-scattering angle variance (exit angle)
    double perpmomvar = 0.0;    // transverse momentum variance     = sum p_i^2 dscat_i
    double perpposvar = 0.0;    // transverse position variance      = sum (L-s_i)^2 dscat_i
    double perpposmomcov = 0.0; // position-momentum covariance      = sum (L-s_i) p_i dscat_i
    double kappa = 0.0;         // Vavilov kappa of the whole crossing (straggling regime)
    double peff = 0.0;          // momentum-integration effective momentum used for scattering
    unsigned nstep = 0;
  };

  template <class KTRAJ> class ElementXing {
    public:
      using PTRAJ = ParticleTrajectory<KTRAJ>;
      using KTRAJPTR = std::shared_ptr<KTRAJ>;
      ElementXing() {}
      virtual ~ElementXing() {}
      // clone op for reinstantiation
      ElementXing(ElementXing const& rhs) = default;
      virtual std::shared_ptr< ElementXing<KTRAJ> > clone(CloneContext&) const;
      virtual void updateReference(PTRAJ const& ptraj) = 0; // update the trajectory reference
      virtual void updateState(MetaIterConfig const& config,bool first) =0; // update the state according to this meta-config
      virtual Parameters params() const =0; // parameter change induced by this element crossing WRT the reference parameters going forwards in time
      virtual double time() const=0; // time the particle crosses thie element
      virtual double transitTime() const=0; // time to cross this element
      virtual KTRAJ const& referenceTrajectory() const =0; // trajectory WRT which the xing is defined
      virtual std::vector<MaterialXing>const&  matXings() const =0; // Effect of each physical material component of this detector element on this trajectory
      virtual void print(std::ostream& ost=std::cout,int detail=0) const =0;
      virtual bool active() const =0;
      virtual bool thickScatter() const { return false; }
      // momentum change and variance increase associated with crossing this element forwards in time, in spatial basis
      void momentumChange(SVEC3& dmom, SMAT& dmomvar) const;
      // full particle-state change and noise covariance associated with crossing this element forwards in time
      void particleStateChange(SVEC3& dmom, DMAT& dstatevar, bool thickscat = false) const;
      // momentum-integrated thick-crossing material effects (sub-stepped, momentum tracked)
      ThickXingEffects thickEffects() const;
      // parameter change associated with crossing this element forwards in time
      Parameters parameterChange(double varscale=1.0) const;
      // cumulative material effect from all the materials in this element crossing going forwards in time, expressed as scalars
      void materialEffects(double& dmom, double& paramomvar, double& perpmomvar) const;
      // sum radiation fraction
      double radiationFraction() const;
    private:
  };

  // cloning requires domain knowledge of pointer members of the cloned object,
  // which must be reassigned explicitly; the default action is thus to throw
  // an error if a clone routine has not been explicitly provided.
  template<class KTRAJ> std::shared_ptr< ElementXing<KTRAJ> > ElementXing<KTRAJ>::clone(CloneContext&) const{
    std::string msg = "Attempt to clone KinKal::ElementXing subclass with no clone() implementation";
    throw std::runtime_error(msg);
  }

  template <class KTRAJ> void ElementXing<KTRAJ>::momentumChange(SVEC3& dmom, SMAT& dmomvar) const {
    // compute the momentum change moving forwards time in global cartesian basis
    // first, compute the change and variance in the momentum basis (along and perp to the momentum direction)
    double dm, paramomvar, perpmomvar;
    materialEffects(dm, paramomvar,perpmomvar);
    // momentum change in forward time direction is due to energy loss; this is along the momentum
    auto momdir = referenceTrajectory().direction(time());
    dmom = dm*SVEC3(momdir.X(),momdir.Y(),momdir.Z());
    // now compute the covariance; this includes smearing from energy straggling and multiple scattering.
    // This is a diagonal matrix in momentum basis; no physical correlation between stragling and scattering, or orthogonal scattering.
    ROOT::Math::SVector<double, 6>  varvec(paramomvar, 0, perpmomvar, 0, 0, perpmomvar);
    SMAT mmvar(varvec); // construct matrix from upper-diagonal elements
    // loop over the momentum change basis directions and create the transform matrix between that and global Cartesian basis
    SSMAT dmdxyz; // momentum basis -> Cartesian conversion matrix
    for(int idir=0;idir<MomBasis::ndir; idir++) {
      auto mdir = referenceTrajectory().direction(time(),static_cast<MomBasis::Direction>(idir));
      SVEC3 vmdir(mdir.X(), mdir.Y(), mdir.Z());
      dmdxyz.Place_in_col(vmdir,0,idir);
    }
    // return covariance in global Cartesian coordinates
    dmomvar = ROOT::Math::Similarity(dmdxyz,mmvar);
  }

  // Sub-step count knobs for the thick crossing.  thickStepRadFrac sets the target radiation
  // fraction per sub-step (so the momentum is ~constant within a step); thickMaxSubSteps caps it.
  inline constexpr double thickStepRadFrac = 0.05;
  inline constexpr unsigned thickMaxSubSteps = 400;

  template <class KTRAJ> ThickXingEffects ElementXing<KTRAJ>::thickEffects() const {
    ThickXingEffects eff;
    double const mass = referenceTrajectory().mass();
    double const p0 = referenceTrajectory().momentum();
    double E = referenceTrajectory().energy();
    double p = p0;
    if(p<=0.0) return eff;

    // total path through all materials of this element: the displacement lever arm is referenced
    // to the element exit plane, so a scatter at global path s contributes through (Ltot - s).
    double Ltot(0.0);
    for(auto const& mxing : matXings()) Ltot += std::abs(mxing.plen_);
    if(Ltot<=0.0) return eff;

    // Sub-step the energy loss with the momentum tracked, accumulating the MEAN (Bethe-Bloch,
    // additive) energy loss and the Gaussian (Bohr) straggling variance.  Also accumulate the
    // momentum-weighted scattering measure sum(dl/(beta p)^2): this is what sets the effective
    // momentum for the multiple-scattering width below.
    double invsum(0.0); // sum of dl/(beta p)^2 over the path
    for(auto const& mxing : matXings()){
      double const plen = std::abs(mxing.plen_);
      if(plen<=0.0) continue;
      double const radfrac = mxing.dmat_.radiationFraction(plen);
      unsigned const nsub = std::min(thickMaxSubSteps,
          std::max(1u, static_cast<unsigned>(std::ceil(radfrac/thickStepRadFrac))));
      double const dl = plen/nsub;
      for(unsigned istep=0; istep<nsub; ++istep){
        double const beta = MatEnv::DetMaterial::particleBeta(p,mass);
        invsum += dl/(beta*beta*p*p);
        // mean energy loss and Gaussian straggling variance for this thin sub-step
        double const de  = mxing.dmat_.energyLossMean(p,dl,mass);     // <0
        double const dev = mxing.dmat_.energyLossGaussVar(p,dl,mass);
        double const dmdE = E/p;
        eff.dE += de;
        eff.dEVar += dev;
        eff.paramomvar += dev*dmdE*dmdE;
        // advance the particle through this sub-step's mean loss
        E += de; // de<0
        double const p2 = E*E - mass*mass;
        p = p2>0.0 ? std::sqrt(p2) : 0.0;
        ++eff.nstep;
        if(p<=0.0) break;
      }
      if(p<=0.0) break;
    }
    eff.dmom = p - p0; // signed total momentum change

    // Scattering magnitude: evaluate the multiple-scattering variance over the FULL path (so the
    // Lynch-Dahl/Highland log term, a function of total thickness, stays correct) but at an
    // EFFECTIVE momentum whose 1/(beta p)^2 equals the path average.  Sub-stepping scatterAngleVar
    // directly would shrink the per-step log term and badly under-count the scattering.
    double const invavg = invsum/Ltot; // <1/(beta p)^2>
    // invert beta^2 p^2 = p^4/(p^2+m^2) = 1/invavg  ->  p^2 = (1 + sqrt(1+4 invavg m^2))/(2 invavg)
    double const peff2 = (1.0 + std::sqrt(1.0 + 4.0*invavg*mass*mass))/(2.0*invavg);
    double const peff = std::sqrt(peff2);
    eff.peff = peff;
    // Use the correct mass-weighted compound scattering moment here (thick path only); for a single
    // element thickScatterCorrection()==1 so this is identical to the bare scatterAngleVar.
    double scatfull(0.0);
    for(auto const& mxing : matXings())
      scatfull += mxing.dmat_.scatterAngleVar(peff,std::abs(mxing.plen_),mass)
                * mxing.dmat_.thickScatterCorrection();
    eff.scatvar = scatfull;
    eff.kappa = matXings().front().dmat_.vavilovKappa(p0,Ltot,mass);

    // Correlated displacement/angle covariance using the PDG closed form over the full path.
    // This is provably equivalent to the Mu2e 2-point Gauss-plane split validated against
    // fine-step Geant4 (rho = sqrt(3)/2), now driven by the momentum-integrated scattering width
    // and the effective momentum:  Var(y) = scatvar x^2/3, Cov(y,p) = scatvar x p/2, Var(p) = scatvar p^2.
    eff.perpmomvar    = scatfull*peff*peff;
    eff.perpposvar    = scatfull*Ltot*Ltot/3.0;
    eff.perpposmomcov = scatfull*Ltot*peff/2.0;
    return eff;
  }

  template <class KTRAJ> void ElementXing<KTRAJ>::particleStateChange(SVEC3& dmom, DMAT& dstatevar, bool thickscat) const {
    auto momdir = referenceTrajectory().direction(time());

    if(!thickscat){
      // thin-scatterer behaviour: momentum-only noise, evaluated at the entrance momentum.
      // This path is bit-identical to the historical implementation (regression gate).
      SMAT dmomvar;
      momentumChange(dmom,dmomvar); // sets dmom (energy loss) and the momentum-block covariance
      dstatevar = DMAT();
      dstatevar.Place_at(dmomvar,3,3);
      return;
    }

    // thick-scatterer: momentum-integrated mean energy loss + correlated scattering covariance
    ThickXingEffects const eff = thickEffects();
    dmom = eff.dmom*SVEC3(momdir.X(),momdir.Y(),momdir.Z());

    dstatevar = DMAT();
    auto addSymOuter = [](DMAT& cov, SVEC6 const& avec, SVEC6 const& bvec, double scale) {
      for(size_t i=0; i<NParams(); ++i){
        for(size_t j=0; j<=i; ++j){
          cov(i,j) += scale*avec[i]*bvec[j];
        }
      }
    };

    SVEC6 momvec(0.0,0.0,0.0,momdir.X(),momdir.Y(),momdir.Z());
    addSymOuter(dstatevar,momvec,momvec,eff.paramomvar);

    for(int idir=1; idir<MomBasis::ndir; ++idir) {
      auto pdir = referenceTrajectory().direction(time(),static_cast<MomBasis::Direction>(idir));
      SVEC6 posvec(pdir.X(),pdir.Y(),pdir.Z(),0.0,0.0,0.0);
      SVEC6 perpmomvec(0.0,0.0,0.0,pdir.X(),pdir.Y(),pdir.Z());
      addSymOuter(dstatevar,posvec,posvec,eff.perpposvar);
      addSymOuter(dstatevar,perpmomvec,perpmomvec,eff.perpmomvar);
      addSymOuter(dstatevar,posvec,perpmomvec,eff.perpposmomcov);
      addSymOuter(dstatevar,perpmomvec,posvec,eff.perpposmomcov);
    }
  }

  template <class KTRAJ> Parameters ElementXing<KTRAJ>::parameterChange(double varscale) const {
    // compute this xing's effect on parameters and covariance. First compute the momentum change and variance
    SVEC3 dmom;
    SMAT dmomvar;
    momentumChange(dmom,dmomvar);
    // convert that to parameter space
    DPDV dPdM = referenceTrajectory().dPardM(time());
    auto dmomp = dPdM*dmom;
    // scale covariance as needed
    dmomvar*= varscale;
    // jacobean transform of covariance into parameter space
    auto dmompvar = ROOT::Math::Similarity(dPdM,dmomvar);
    return Parameters(dmomp,dmompvar);
  }

  template <class KTRAJ> void ElementXing<KTRAJ>::materialEffects(double& dmom, double& paramomvar, double& perpmomvar) const {
    // accumulate the change in energy and scattering angle variance from the material components
    double E = referenceTrajectory().energy();
    double mom = referenceTrajectory().momentum();
    double mass = referenceTrajectory().mass();
    // loop over individual materials and accumulate their effects
    double dE(0.0), dEVar(0.0), scatvar(0.0);
    for(auto const& mxing : matXings()){
      dE += mxing.dmat_.energyLoss(mom,mxing.plen_,mass);
      dEVar += mxing.dmat_.energyLossVar(mom,mxing.plen_,mass);
      scatvar += mxing.dmat_.scatterAngleVar(mom,mxing.plen_,mass);
    }
    // convert energy change to change in momentum
    double dmdE = E/mom;
    dmom = dE*dmdE;
    paramomvar = dEVar*dmdE*dmdE;
    // scattering applies directly to momentum (1st order)
    perpmomvar = scatvar*mom*mom;
  }

  template <class KTRAJ> double ElementXing<KTRAJ>::radiationFraction() const {
    double retval(0.0);
    for(auto const& mxing : matXings())
      retval += mxing.dmat_.radiationFraction(mxing.plen_);
    return retval;
  }

}
#endif
