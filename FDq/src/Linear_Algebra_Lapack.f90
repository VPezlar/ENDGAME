!-----------------------------------------------------------------------!
!                            Linear_Algebra                             !
!                                                                       !
!                                                                       !
!--------------------------------------------Miguel Hermanns-2005.02.30-!
module Linear_Algebra
implicit none

contains
!-----------------------------------------------------------------------!
subroutine Linear_Algebraic_System(Matrix, Constants, Variables)
real, intent(in)  :: Matrix(:,:), Constants(:)
real, intent(out) :: Variables(:)

integer    :: Number_Equations, Pivoting(ubound(Matrix,1)), info
real       :: Array(ubound(Matrix,1),ubound(Matrix,2))

external DGESV

Number_Equations = ubound(Matrix,1)
Array            = Matrix
Variables        = Constants
  
call DGESV(Number_Equations, 1, Array, Number_Equations, Pivoting, &
           Variables, Number_Equations, info)

if(info /= 0) write(*,*) "LAPACK Error: ", info

end subroutine Linear_Algebraic_System
!!!-----------------------------------------------------------------------!
!!pure subroutine Gauss_Elimination(Matrix, Constants, Variables)
!!real, intent(in)  :: Matrix(:,:), Constants(:)
!!real, intent(out) :: Variables(:)
!
!integer :: Number_Equations, i, j
!real    :: C(ubound(Matrix,1),ubound(Matrix,2)+1)
!
!!Extracts the information of the shapes
!Number_Equations = ubound(Matrix, 1)
!
!!Constructs the augmented matrix C
!C(:,1:Number_Equations) = Matrix
!C(:,Number_Equations+1) = Constants
!
!!Solves the system of equations by Gauss elimination
!do j = 1, Number_Equations
!
!   C(j,:) = C(j,:) / C(j,j)
!
!   do i = 1, Number_Equations
!      if (i /= j) C(i,:) = C(i,:) - C(i,j) * C(j,:)
!   enddo
!enddo
!
!Variables = C(:,Number_Equations+1)

!end subroutine Gauss_Elimination
!end subroutine Linear_Algebraic_System
!-----------------------------------------------------------------------!
end module Linear_Algebra
