#ifndef KinKal_Hit_hh
#define KinKal_Hit_hh
//
//  Base class to describe a measurement that constrains some aspect of the track fit
//  The constraint is expressed as a weight WRT a set of reference parameters.
//
#include "KinKal/General/CloneContext.hh"
#include "KinKal/General/Weights.hh"
#include "KinKal/General/Parameters.hh"
#include "KinKal/General/Chisq.hh"
#include "KinKal/Trajectory/ParticleTrajectory.hh"
#include "KinKal/Fit/MetaIterConfig.hh"
#include <memory>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <string>

namespace KinKal {
  template <class KTRAJ> class Hit {
    public:
      using PTRAJ = ParticleTrajectory<KTRAJ>;
      using KTRAJPTR = std::shared_ptr<KTRAJ>;
      Hit() {}
      virtual ~Hit(){}
      // disallow copy and equivalence
      Hit(Hit const& ) = delete;
      Hit& operator =(Hit const& ) = delete;
      // clone op for reinstantiation
      virtual std::shared_ptr< Hit<KTRAJ> > clone(CloneContext&) const;
     // hits may be active (used in the fit) or inactive; this is a pattern recognition feature
      virtual bool active() const =0;
      virtual unsigned nDOF() const=0;
      virtual Chisq chisq(Parameters const& params) const =0;  // least-squares distance to given parameters
      virtual double time() const = 0;  // time of this hit: this is WRT the reference trajectory
      virtual void print(std::ostream& ost=std::cout,int detail=0) const = 0;
      // update to a new reference, without changing internal state
      virtual void updateReference(PTRAJ const& ptraj) = 0;
      virtual KTRAJPTR refTrajPtr() const = 0;
      // update the internals of the hit, specific to this meta-iteraion
      virtual void updateState(MetaIterConfig const& config,bool first) = 0;
      // The following provides the constraint/information content of this hit in the trajectory weight space
      virtual Weights const& weight() const = 0;
      KTRAJ const& referenceTrajectory() const { return *refTrajPtr(); }  // trajectory WRT which the weight etc is defined
      // parameters WRT which this hit's residual and weights are set.  These are generally biased
      // in that they contain the information of this hit
      Parameters const& referenceParameters() const { return referenceTrajectory().params(); }
      // record the parameters of the piece of a new fit trajectory that holds this hit's weight
      void setWeightParameters(Parameters const& wpars) { wpars_ = wpars; }
      // Unbiased parameters WRT the reference: this hit's weight removed from the piece it was applied to.  These describe
      // the fit that produced the current reference; the hit's state was decided one fit cycle earlier, against the
      // previous reference
      Parameters unbiasedParameters() const;
      // unbiased least-squares distance to reference parameters
      Chisq chisquared() const;
    private:
      std::optional<Parameters> wpars_; // parameters of the piece the weight was applied to; empty until set (eg cloned hits)
  };

  // cloning requires domain knowledge of pointer members of the cloned object,
  // which must be reassigned explicitly; the default action is thus to throw
  // an error if a clone routine has not been explicitly provided.
  template<class KTRAJ> std::shared_ptr< Hit<KTRAJ> > Hit<KTRAJ>::clone(CloneContext& context) const{
    std::string msg = "Attempt to clone KinKal::Hit subclass with no clone() implementation";
    throw std::runtime_error(msg);
  }

  template<class KTRAJ> Parameters Hit<KTRAJ>::unbiasedParameters() const {
    if(active()){
      // remove this hit's weight from the piece it was applied to: removing it from another piece, separated by
      // material effects, over-subtracts and can leave a covariance that isn't positive definite.  Without a
      // recorded piece, the weight is taken as applied to the reference
      Parameters const& wpars = wpars_ ? *wpars_ : referenceParameters();
      Weights wt(wpars);
      wt -= weight();
      Parameters upars(wt);
      // express the result WRT the reference: apply the parameter change from removing this hit to the reference
      // parameters, keeping the unbiased covariance.  Identical to the plain subtraction when the pieces coincide
      upars.parameters() += referenceParameters().parameters() - wpars.parameters();
      return upars;
    } else
      return referenceParameters();
  }

  template<class KTRAJ> Chisq Hit<KTRAJ>::chisquared() const {
    if(active()){
      return chisq(unbiasedParameters());
    } else
      return Chisq();
  }

  template <class KTRAJ> std::ostream& operator <<(std::ostream& ost, Hit<KTRAJ> const& thit) {
    thit.print(ost,0);
    return ost;
  }

}
#endif

