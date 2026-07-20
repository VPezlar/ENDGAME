!-----------------------------------------------------------------------!
!                     Lagrange_Interpolation_Method                     !
!                                                                       !
!                                                                       !
!                                                                       !
!                                                                       !
! History of the module:                                                !
!                                                                       !
!    Date    | Version | Comments                                       !
! --------------------------------------------------------------------- !
! 2004.12.15 |   0.0   | Creation of the module.                        !
!            |         |                                                !
!                                                                       !
!--------------------------------------------Miguel Hermanns-2004.12.15-!
module Lagrange_Interpolation_Method
implicit none

contains
!-----------------------------------------------------------------------!
!                         Lagrange_Coefficients                         !
!                                                                       !
! Given a set of nodes {x_j, j = 0,...,N} and a target point x, this    !
! subroutine computes the coefficients for the Lagrange interpolation   !
! formula                                                               !
!                                 N                                     !
!                                ---                                    !
!                         I(x) = >   L (x) f(x )                        !
!                                ---  j       j                         !
!                                j=0                                    !
!                                                                       !
! as well as the coefficients for the derivatives and for the integral  !
! of the interpolant. The computation is done using the following nes-  !
! ted expression for the derivatives of the Lagrange polynomials:       !
!                                                                       !
!        k                /    k-1                    k        \        !
!       d L_j,r      1    |   d   L_j,r-1            d L_j,r-1 |        !
!       ------- = ------- | k ----------- + (x - x ) --------- |        !
!           k      x - x  |        k-1            r       k    |        !
!         dx        j   r \      dx                     dx     /        !
!                                                                       !
! where L_j,0(x) = 1 and r = 0,...,j-1,j+1,...,N, hence L_j = L_j,N.    !
! For the computation of the integral of the Lagrange polynomials, the  !
! following expression for the definite integral is used:               !
!                                                                       !
!                  x                                                    !
!                 /             N         k      k+1                    !
!                 |            ---     k d L_j  x                       !
!                 | L (x) dx = >   (-1)  ----- ------                   !
!                 |  j         ---          k  (k+1)!                   !
!                 /            k=0        dx                            !
!                0                                                      !
!                                                                       !
! Input:                                                                !
!    Nodes(0:N): Nodes to be used by the Lagrange interpolant.          !
!    Point: Point in which to evaluate the Lagrange interpolant.        !
!    Derivatives(1:Nd): Derivatives to compute. In order to ask for     !
!                       the integral, a -1 value has to be supplied.    !
!                                                                       !
! Output:                                                               !
!    Coefficients(1:Nd,0:N): Coefficients of the Lagrange interpolant,  !
!                            of the requested derivatives and integral. !
!                                                                       !
!--------------------------------------------Miguel Hermanns-2004.12.15-!
pure subroutine Lagrange_Coefficients(Nodes, Point, Derivatives, Coefficients)
real, intent(in)    :: Nodes(0:), Point
integer, intent(in) :: Derivatives(:)
real, intent(out)   :: Coefficients(:,0:)

integer :: Number_Nodes, Lowest_Derivative, Highest_Derivative, j, r, k
real    :: L_j(-1:size(Nodes)), dx_r, dx_jr, factorial

!Extracts the information of the shapes
Number_Nodes       = ubound(Nodes,    1)
Lowest_Derivative  = minval(Derivatives)
Highest_Derivative = maxval(Derivatives)

!Computes all the derivatives, if the first integral has to be computed
if(Lowest_Derivative == -1) Highest_Derivative = Number_Nodes

!Computes each of the Lagrange coefficients
do j = 0, Number_Nodes

  L_j    = 0d0
  L_j(0) = 1d0

  do r = 0, Number_Nodes

    if(r == j) cycle

    dx_r  = Point    - Nodes(r)
    dx_jr = Nodes(j) - Nodes(r)

    do k = Highest_Derivative, 0, -1
      L_j(k) = 1d0 / dx_jr * (k * L_j(k-1) + dx_r * L_j(k))
    enddo
  enddo

  if(Lowest_Derivative == -1) then
    factorial = 1d0

    do k = 0, Highest_Derivative
      factorial = factorial * (k + 1)
      L_j(-1)   = L_j(-1) + (-1d0)**k * L_j(k) * Point**(k+1) / factorial
    enddo
  endif

  Coefficients(:,j) = L_j(Derivatives)
enddo

end subroutine Lagrange_Coefficients
!-----------------------------------------------------------------------!
!                          Lagrange_Interpolant                         !
!                                                                       !
!                                                                       !
!                                 N                                     !
!                                ---                                    !
!                         I(x) = >   L (x) f(x )                        !
!                                ---  j       j                         !
!                                j=0                                    !
!                                                                       !
!                        k        N   k                                 !
!                       d I      --- d L_j                              !
!                       ---(x) = >   -----(x) f(x )                     !
!                         k      ---    k        j                      !
!                       dx       j=0  dx                                !
!                                                                       !
!                     x                x                                !
!                    /            N   /                                 !
!                    |           ---  |                                 !
!                    | I(x) dx = >    | L (x) dx f(x )                  !
!                    |           ---  |  j          j                   !
!                    /           k=0  /                                 !
!                   0                0                                  !
!                                                                       !
! Input:                                                                !
!    Coefficients(1:Nd,0:N):                                            !
!    Values(0:N):                                                       !
!                                                                       !
! Output:                                                               !
!    Interpolant(1:Nd):                                                 !
!                                                                       !
!--------------------------------------------Miguel Hermanns-2004.11.29-!
pure subroutine Lagrange_Interpolant(Coefficients, Values, Interpolant)
real, intent(in)  :: Coefficients(:,0:), Values(0:)
real, intent(out) :: Interpolant(:)

Interpolant = matmul(Coefficients,Values)

end subroutine Lagrange_Interpolant
!-----------------------------------------------------------------------!
!                       Lagrange_Error_Polynomial                       !
!                                                                       !
! Given a set of nodes {x_j, j = 0,...,N} and a target point x, this    !
! subroutine computes the following error polynomial                    !
!                                                                       !
!                                  N                                    !
!                                 ---                                   !
!                         TT(x) = | | (x - x )                          !
!                                 j=0       j                           !
!                                                                       !
! as well as its derivatives and intergral at the specified point x.    !
! The algorithm for computing the derivatives and the integral is the   !
! same as the one for the computation of the Lagrange coefficients in   !
! this same module.                                                     !
!                                                                       !
! The resulting interpolation error associated with the Lagrange in-    !
! terpolation formula will be given by:                                 !
!                                                                       !
!                              TT(x)   ,(N+1)                           !
!                      E(x) = ------- f      (xi)                       !
!                             (N+1)!                                    !
!                                                                       !
! where xi is an unknown point in the interval [x_0, x_N]. The k-th de- !
! rivative of the interpolation error will be given by:                 !
!                                                                       !
!                   k                                                   !
!        ,(k)      ---        k!         ,(i)     ,(N+k-i+1)            !
!       E    (x) = >   --------------- TT    (x) f          (xi )       !
!                  ---  (N+k-i+1)! i!                          i        !
!                  i=0                                                  !
!                                                                       !
! where xi_i are unknown points in the interval [x_0, x_N].             !
!                                                                       !
! Input:                                                                !
!    Nodes(0:N): Nodes used by the Lagrange interpolantion.             !
!    Point: Point is which to evaluate the Lagrange error polynomial.   !
!    Derivatives(1:Nd): Derivatives to compute. In order to ask for     !
!                       the integral, a -1 value has to be supplied.    !
!                                                                       !
! Output:                                                               !
!    Error_Polynomial(1:Nd): Requested derivatives and integral of the  !
!                            Lagrange error polynomial.                 !
!                                                                       !
!--------------------------------------------Miguel Hermanns-2005.02.14-!
pure subroutine Lagrange_Error_Polynomial(Nodes, Point, Derivatives, &
                                          Error_Polynomial)
real, intent(in)    :: Nodes(0:), Point
integer, intent(in) :: Derivatives(:)
real, intent(out)   :: Error_Polynomial(:)

integer :: Number_Nodes, Lowest_Derivative, Highest_Derivative, r, k
real    :: pi(-1:size(Nodes)), factorial

!Extracts the information of the shapes
Number_Nodes       = ubound(Nodes, 1)
Lowest_Derivative  = minval(Derivatives)
Highest_Derivative = maxval(Derivatives)

!Computes all the derivatives, if the first integral has to be computed
if(Lowest_Derivative == -1) Highest_Derivative = Number_Nodes

!Computes the Lagrange Error pi(x), its derivatives and its integral
pi    = 0d0
pi(0) = 1d0

do r = 0, Number_Nodes
  do k = Highest_Derivative, 0, -1
    pi(k) = k * pi(k-1) + (Point - Nodes(r)) * pi(k)
  enddo
enddo

if(Lowest_Derivative == -1) then

  factorial = 1d0

  do k = 0, Highest_Derivative
    factorial = factorial * (k + 1)
    pi(-1)    = pi(-1) + (-1d0)**k * pi(k) * Point**(k+1) / factorial
  enddo

endif

Error_Polynomial = pi(Derivatives)

end subroutine Lagrange_Error_Polynomial
!-----------------------------------------------------------------------!
end module Lagrange_Interpolation_Method
