#ifndef IGL_COPYLEFT_CGAL_FAST_WINDING_NUMBER
#define IGL_COPYLEFT_CGAL_FAST_WINDING_NUMBER
#include "../../igl_inline.h"
#include <Eigen/Core>
#include <vector>
namespace igl
{
  namespace copyleft
  {
    namespace cgal
    {
      /// Evaluate the fast winding number for point data, without known areas. The
      /// areas are calculated using igl::knn and igl::copyleft::cgal::point_areas.
      ///
      /// This function performes the precomputation and evaluation all in one.
      /// If you need to acess the precomuptation for repeated evaluations, use the
      /// two functions designed for exposed precomputation, which are the first two
      /// functions see in igl/fast_winding_number.h
      ///
      /// @param[in] P  #P by 3 list of point locations
      /// @param[in] N  #P by 3 list of point normals
      /// @param[in] Q  #Q by 3 list of query points for the winding number
      /// @param[in] expansion_order    the order of the taylor expansion. We support 0,1,2.
      /// @param[in] beta  This is a Barnes-Hut style accuracy term that separates near feild
      ///         from far field. The higher the beta, the more accurate and slower
      ///         the evaluation. We reccommend using a beta value of 2.
      /// @param[out] WN  #Q by 1 list of windinng number values at each query point
      ///
      template <
        typename DerivedP, 
        typename DerivedN, 
        typename DerivedQ,
        typename BetaType, 
        typename DerivedWN>
      IGL_INLINE void fast_winding_number(
        const Eigen::MatrixBase<DerivedP>& P,
        const Eigen::MatrixBase<DerivedN>& N,
        const Eigen::MatrixBase<DerivedQ>& Q,
        const int expansion_order,
        const BetaType beta,
        Eigen::PlainObjectBase<DerivedWN>& WN);
      /// \overload
      template <
        typename DerivedP, 
        typename DerivedN, 
        typename DerivedQ, 
        typename DerivedWN>
      IGL_INLINE void fast_winding_number(
        const Eigen::MatrixBase<DerivedP>& P,
        const Eigen::MatrixBase<DerivedN>& N,
        const Eigen::MatrixBase<DerivedQ>& Q,
        Eigen::PlainObjectBase<DerivedWN>& WN);
    }
  }
}
#ifndef IGL_STATIC_LIBRARY
#  include "fast_winding_number.cpp"
#endif

#endif

