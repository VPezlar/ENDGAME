!-----------------------------------------------------------------------!
!                         Optimum_Grid_Methods                          !
!                                                                       !
!                                                                       !
! History of the module:                                                !
!                                                                       !
!    Date    | Version | Comments                                       !
! --------------------------------------------------------------------- !
! 2004.11.29 |   0.0   | Creation of the module.                        !
!            |         |                                                !
!                                                                       !
!--------------------------------------------Miguel Hermanns-2005.03.11-!
module Optimum_Grid_Methods
use Lagrange_Interpolation_Method
use Piecewise_Polynomial_Interpolation, only: Centered_Polynomials
use NonLinear_Algebra
implicit none

integer Degree

contains
!-----------------------------------------------------------------------!
!                          Optimum_Grid_Nodes                           !
!                                                                       !
!                                                                       !
!--------------------------------------------Miguel Hermanns-2005.02.14-!
subroutine Optimum_Grid_Nodes(Polynomial_Degree, Nodes, Dual_Nodes)
integer, intent(in) :: Polynomial_Degree
real, intent(out)   :: Nodes(0:), Dual_Nodes(:)

optional :: Dual_Nodes

integer :: Number_Nodes, i
real    :: Gauss_Nodes(1:ubound(Nodes,1))

!Extracts the information of the shapes
Number_Nodes = ubound(Nodes, 1)

!Specifies a uniform mesh as initial guess
do i = 1, Number_Nodes
  Gauss_Nodes(i) = -1d0 + 2d0 * real(2 * i - 1) / real(2 * Number_Nodes)
enddo

!Solves increasing polynomial degrees to ensure convergence
do Degree = 1, Polynomial_Degree-1, 2
  call Newton_Raphson_Method(Nonlinear_System = Gauss_Extrema_Conditions, &
                             Variables        = Gauss_Nodes               )
enddo

!Computes the dual nodes of the Gauss nodes, which are the interesting ones
call Extrema_Nodes(Polynomial_Degree-1, Gauss_Nodes, Nodes)

if(present(Dual_Nodes)) Dual_Nodes = Gauss_Nodes

end subroutine Optimum_Grid_Nodes

!-----------------------------------!
subroutine Gauss_Extrema_Conditions(Nodes, Equations)
real, intent(in)  :: Nodes(0:)
real, intent(out) :: Equations(0:)

integer :: Number_Nodes, i
real    :: Extrema(0:ubound(Nodes,1)+1), Extrema_Errors(0:ubound(Nodes,1)+1,0:2)           

!Extracts the information of the shapes
Number_Nodes = ubound(Nodes, 1)

!Determines the extrema of pi(x) in each domain of validity
call Extrema_Nodes(Degree, Nodes, Extrema, Extrema_Errors)

!Constructs the equations
do i = 0, Number_Nodes
  Equations(i) = abs(Extrema_Errors(i+1,0)) - abs(Extrema_Errors(i,0))
enddo

end subroutine Gauss_Extrema_Conditions
!-----------------------------------!
subroutine Extrema_Nodes(Polynomial_Degree, Nodes, Extrema, Extrema_Errors)
integer, intent(in) :: Polynomial_Degree
real, intent(in)    :: Nodes(0:)
real, intent(out)   :: Extrema(0:), Extrema_Errors(0:,0:)

optional :: Extrema_Errors

integer :: Number_Nodes, Seeds(ubound(Nodes,1)), i, s, q, Index, &
           Stencils(ubound(Nodes,1)), Derivatives(3)
real    :: Iteration_Error, Errors(0:2)

!Extracts the information of the shapes
Number_Nodes = ubound(Nodes,   1)

!Assigns a centered polynomial to each domain of validity
call Centered_Polynomials(Polynomial_Degree, Seeds, Stencils)

!Determines the extrema of pi(x) in each domain of validity
Extrema(0)              = -1d0
Extrema(Number_Nodes+1) = 1d0
Extrema(1:Number_Nodes) = (Nodes(1:Number_Nodes) + Nodes(0:Number_Nodes-1)) / 2d0

do i = 0, Number_Nodes+1
  
  Index = max(1, min(i, Number_Nodes))
  s     = Seeds(   Index)
  q     = Stencils(Index) - 1

  Derivatives     = (/ 0, 1, 2 /)
  Iteration_Error = 1d0

  call Lagrange_Error_Polynomial(Nodes(s:s+q), Extrema(i), Derivatives, Errors(:))
 
  if((i /= 0).and.(i /= Number_Nodes+1)) then
    do while (Iteration_Error > 1d-12)

      call Lagrange_Error_Polynomial(Nodes(s:s+q), Extrema(i), Derivatives, Errors(:))

      Extrema(i)      = Extrema(i) - Errors(1) / Errors(2)
      Iteration_Error = abs(Errors(1) / Errors(2))
    enddo
  endif

  if(present(Extrema_Errors)) Extrema_Errors(i,:) = Errors
enddo

end subroutine Extrema_Nodes
!-----------------------------------!

!-----------------------------------------------------------------------!
end module Optimum_Grid_Methods
