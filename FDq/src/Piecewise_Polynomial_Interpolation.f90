!-----------------------------------------------------------------------!
!                   Piecewise_Polynomial_Interpolation                  !
!                                                                       !
! See the following reference for mathematical details of the implemen- !
! ted interpolation algorithm:                                          !
!                                                                       !
! Hermanns M & Hernandez JA, 2008: "Stable high-order finite-difference !
! methods based on non-uniform grid point distributions", International !
! Journal for Numerical Methods in Fluids, Volume 56, pages 233-255.    !
!                                                                       !
! History of the module:                                                !
!                                                                       !
!    Date    | Version | Comments                                       !
! --------------------------------------------------------------------- !
! 2004.11.29 |   0.0   | Creation of the module.                        !
!            |         |                                                !
!                                                                       !
!--------------------------------------------Miguel Hermanns-2011.01.05-!
module Piecewise_Polynomial_Interpolation
use Lagrange_Interpolation_Method
implicit none

private
public :: Interpolate, Interpolation_Matrix, Polynomial_Degree, &
          Centered_Polynomials

!Degree of the polynomials to use
integer :: Polynomial_Degree = 1

contains
!-----------------------------------------------------------------------!
!                             Interpolate                               !
!                                                                       !
! Given a set of grid nodes and nodal values, the present subroutine    !
! interpolates these values onto a given set of interpolation points    !
! using a piecewise polynomial interpolant.                             !
!                                                                       !
! The degree of the polynomials is specified through the global varia-  !
! ble Polynomial_Degree defined in the header of the present module and !
! which is accesible through host association.                          !
!                                                                       !
! The present subroutine not only returns the value of the interpolant  !
! at the specified interpolation points, but also its derivatives, if   !
! requested so by the user.                                             !
!                                                                       !
! Input:                                                                !
!    Grid_Nodes(0:N):                                                   !
!    Nodal_Values(0:N):                                                 !
!    Interpolation_Points(0:Np):                                        !
!    Derivatives(1:Nd):                                                 !
!                                                                       !
! Output:                                                               !
!    Interpolated_Values(0:Np,1:Nd):                                    !
!                                                                       !
!--------------------------------------------Miguel Hermanns-2011.01.05-!
pure subroutine Interpolate(Grid_Nodes, Nodal_Values, Interpolation_Points, &
                            Derivatives, Interpolated_Values)
real,    intent(in)  :: Grid_Nodes(0:), Nodal_Values(0:), &
                        Interpolation_Points(0:)
integer, intent(in)  :: Derivatives(:)
real,    intent(out) :: Interpolated_Values(0:,:)

integer :: Number_Nodes, Number_Points, p, s, q,  &
           Domain_Seeds(   ubound(Grid_Nodes,1)), &
           Domain_Stencils(ubound(Grid_Nodes,1)), &
           Mapping(0:ubound(Interpolation_Points,1))
real    :: Coefficients(ubound(Derivatives,1),0:Polynomial_Degree)

!Extracts the information of the shapes
Number_Nodes  = ubound(Grid_Nodes,           1)
Number_Points = ubound(Interpolation_Points, 1)

!Associates the interpolation points with the domains of validity
call Associate_Points_with_Domains(Grid_Nodes, Interpolation_Points, Mapping)

!Assigns a centered polynomial to each domain of validity
call Centered_Polynomials(Polynomial_Degree, Domain_Seeds, Domain_Stencils)

!Interpolates the supplied values at the specified interpolation points
do p = 0, Number_Points

  s = Domain_Seeds(   Mapping(p))
  q = Domain_Stencils(Mapping(p)) - 1

  !Computes the Lagrange coefficients at the interpolation points
  call Lagrange_Coefficients(Nodes        = Grid_Nodes(s:s+q),       &
                             Point        = Interpolation_Points(p), &
                             Derivatives  = Derivatives,             &
                             Coefficients = Coefficients(:,0:q)      )

  !Evaluates the piecewise polynomial interpolant
  call Lagrange_Interpolant(Coefficients = Coefficients(:,0:q),    &
                            Values       = Nodal_Values(s:s+q),    &
                            Interpolant  = Interpolated_Values(p,:))
enddo

end subroutine Interpolate
!-----------------------------------------------------------------------!
!                        Interpolation_Matrix                           !
!                                                                       !
!                                                                       !
! Input:                                                                !
!    Grid_Nodes(0:N):                                                   !
!    Interpolation_Points(0:Np):                                        !
!    Derivatives(1:Nd):                                                 !
!                                                                       !
! Output:                                                               !
!    Matrix(0:Np,1:Nd,0:N):                                             !
!                                                                       !
!--------------------------------------------Miguel Hermanns-2011.01.05-!
pure subroutine Interpolation_Matrix(Grid_Nodes, Interpolation_Points, &
                                     Derivatives, Matrix)
real,    intent(in)  :: Grid_Nodes(0:), Interpolation_Points(0:)
integer, intent(in)  :: Derivatives(:)
real,    intent(out) :: Matrix(0:,:,0:)

integer :: Number_Nodes, Number_Points, p, s, q,  &
           Domain_Seeds(   ubound(Grid_Nodes,1)), &
           Domain_Stencils(ubound(Grid_Nodes,1)), &
           Mapping(0:ubound(Interpolation_Points,1))
real    :: Coefficients(ubound(Derivatives,1),0:Polynomial_Degree)

!Extracts the information of the shapes
Number_Nodes  = ubound(Grid_Nodes,           1)
Number_Points = ubound(Interpolation_Points, 1)

!Associates the interpolation points with the domains of validity
call Associate_Points_with_Domains(Grid_Nodes, Interpolation_Points, Mapping)

!Assigns a centered polynomial to each domain of validity
call Centered_Polynomials(Polynomial_Degree, Domain_Seeds, Domain_Stencils)

!Obtains the Lagrange coefficients and constructs with them the matrix
Matrix = 0d0

do p = 0, Number_Points

  s = Domain_Seeds(   Mapping(p))
  q = Domain_Stencils(Mapping(p)) - 1

  !Computes the Lagrange coefficients at the interpolation points
  call Lagrange_Coefficients(Nodes        = Grid_Nodes(s:s+q),       &
                             Point        = Interpolation_Points(p), &
                             Derivatives  = Derivatives,             &
                             Coefficients = Coefficients(:,0:q)      )

  !Constructs the interpolation matrix out of the coefficients
  Matrix(p,:,s:s+q) = Coefficients(:,0:q)

enddo

end subroutine Interpolation_Matrix
!-----------------------------------------------------------------------!
!                         Centered_Polynomials                          !
!                                                                       !
! Given a global degree for the piecewise polynomial interpolation, the !
! present subroutine associates one specific polynomial to each of the  !
! domains of validity by supplying its seed and its stencil.            !
!                                                                       !
! The definitions for the different terms arising in piecewise polyno-  !
! mial interpolations are the following:                                !
!                                                                       !
! shift: distance in index space between the node x_j and the leftmost  !
!        node used by the polynomial associated with the domain of va-  !
!        lidity D_j = (x_j-1,x_j].                                      !
!                                                                       !
! seed: index of the leftmost node used by the polynomial associated    !
!       with the domain of validity D_j.                                !
!                                                                       !
! stencil: total number of nodes used by the polynomial associated with !
!          the domain of validity D_j.                                  !
!                                                                       !
! degree: degree of the polynomial associated with the domain of vali-  !
!         dity D_j.                                                     !
!                                                                       !
! An example is given to clarify the ideas:                             !
!                                                                       !
! x_j-4   x_j-3   x_j-2   x_j-1    x_j    x_j+1   x_j+2   x_j+3   x_j+4 !
!   |-------|-------|-------|-------|-------|-------|-------|-------|   !
!                                                                       !
!                              D_j                                      !
!           |---------------|-------|-----------------------|           !
!                                                                       !
!                                                                       !
! In the shown example, the shift of the polynomial associated with the !
! domain of validity D_j is 3, the seed is j-3, the stencil is 7 and    !
! the degree is 6.                                                      !
!                                                                       !
! The implemented algorithm chooses centered polynomials with respect   !
! to each domain of validity, except close to the boundaries, where the !
! polynomials are biased inwards in order to use only real nodes.       !
!                                                                       !
! If the supplied global degree for the piecewise polynomial interpola- !
! tion is odd (i.e. 1, 3, 5, etc.), perfectly centered polynomials are  !
! obtained with respect to each domain of validity. If even global de-  !
! grees are given instead (i.e. 2, 4, 6, etc.), then the polynomials    !
! are slightly biased to the right, like in the shown example, leading  !
! to polynomials which are perfectly centered with respect to the       !
! rightmost node, x_j, of the domain of validity.                       !
!                                                                       !
! Input:                                                                !
!    Polynomial_Degree: global degree of the piecewise polynomial in-   !
!                       terpolation.                                    !
!                                                                       !
! Output:                                                               !
!    Seeds(1:N): seeds of the polynomials associated to each of the     !
!                domains of validity.                                   !
!    Stencils(1:N): stencils of the polynomials associated to each of   !
!                   the domains of validity.                            !
!                                                                       !
!--------------------------------------------Miguel Hermanns-2004.12.13-!
pure subroutine Centered_Polynomials(Polynomial_Degree, Seeds, Stencils)
integer, intent(in)  :: Polynomial_Degree
integer, intent(out) :: Seeds(:), Stencils(:)

integer :: Number_Domains, Shift, D1, D2, i

!Extracts the information of the shapes
Number_Domains = size(Seeds)

!Determines the limiting domains not affected by the boundaries
Shift = (Polynomial_Degree + 1) / 2  !Uses the integer property 3 / 2 = 1

D1 = max(1, Shift)
D2 = Number_Domains - (Polynomial_Degree - Shift)

!Assigns the seeds and the stencils to each of the domains
Stencils = Polynomial_Degree + 1

                Seeds(:D1-1) = 0
forall(i=D1:D2) Seeds(i)     = i - Shift
                Seeds(D2+1:) = Number_Domains - Polynomial_Degree

end subroutine Centered_Polynomials
!-----------------------------------------------------------------------!
!                    Associate_Points_with_Domains                      !
!                                                                       !
! Given a set of grid nodes {x_j, j = 0,...,N} and a set of interpola-  !
! tion points {p_i, i = 0,...,Np}, the present subroutine determines in !
! which domain of validity D_j each point p_i lies. Each domain of va-  !
! lidity is identified by the index of its upper bound, e.g.:           !
!                                                                       !
!                          D_j = (x_j-1, x_j]                           !
!                                                                       !
! Hence, domain indices go from 1 to N. Points outside of the total     !
! interval [x_0, x_N] are assigned to the domains D_1 and D_N in order  !
! to avoid the existence of unassociated points.                        !
!                                                                       !
! The following rules need to be satisfied by the supplied data to en-  !
! sure the correct working of the algorithm:                            !
!                                                                       !
! 1) Grid nodes x_j and interpolation points p_i must be sequentially   !
!    ordered and satisfy the following relations:                       !
!                                                                       !
!        x_0 < x_1 < ... < x_N         p_0 <= p_1 <= ... <= p_Np        !
!                                                                       !
! 2) If an interpolation point and a grid node are equal, (p_i = x_j),  !
!    then the algorithm associates that point to the left domain:       !
!                                                                       !
!                        p_i c D_j = (x_j-1, x_j]                       !
!                                                                       !
! 3) If two consecutive interpolation points coincide with a grid node  !
!    (p_i-1 = p_i = x_j), then point p_i-1 is associated to the left    !
!    domain and point p_i to the right domain:                          !
!                                                                       !
!    p_i-1 c D_j = (x_j-1, x_j]    and    p_i c D_j+1 = (x_j, x_j+1]    !
!                                                                       !
!                                                                       !
! Input:                                                                !
!    Nodes(0:N): grid nodes to be used by the piecewise polynomial in-  !
!                terpolation.                                           !
!    Points(0:Np): interpolation points on which to evaluate the piece- !
!                  wise polynomial interpolation.                       !
!                                                                       !
! Output:                                                               !
!    Mapping(0:Np): Index of the domain of validity to which each in-   !
!                   terpolation point is associated.                    !
!                                                                       !
!--------------------------------------------Miguel Hermanns-2004.12.13-!
pure subroutine Associate_Points_with_Domains(Nodes, Points, Mapping)
real, intent(in)     :: Nodes(0:), Points(0:)
integer, intent(out) :: Mapping(0:)

integer :: Number_Domains, Number_Points, D_j, i

!Extracts the information of the shapes
Number_Domains = ubound(Nodes,  1)
Number_Points  = ubound(Points, 1)

!Goes over all the interpolation points
D_j = 1

do i = 0, Number_Points

  do while(Points(i) > Nodes(D_j) .and. D_j < Number_Domains)
    D_j = D_j + 1
  enddo

  Mapping(i) = D_j

  if(Points(i) == Nodes(D_j) .and. D_j < Number_Domains) D_j = D_j + 1
enddo

end subroutine Associate_Points_with_Domains
!-----------------------------------------------------------------------!
end module Piecewise_Polynomial_Interpolation
