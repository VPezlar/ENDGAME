SUBROUTINE STD(Nr, qdegree, xcor, DD1, DD2, flag)
   USE :: Optimum_Grid_Methods
   USE :: Piecewise_Polynomial_Interpolation

   IMPLICIT NONE
   INTEGER :: infextlowq 
   integer, intent(in) :: Nr,qdegree, flag
   real, intent(out) :: DD1(Nr, Nr), DD2(Nr, Nr)
   real, intent(inout) :: xcor(Nr)

   integer j,i,jcor,icor,ierr,derv(2)
   real d(Nr,2,Nr),d1(Nr,2,Nr),d2(Nr,2,Nr),d3(Nr,2,Nr)
   real d4(Nr,2,Nr),d5(Nr,2,Nr),d6(Nr,2,Nr)
   real res, xLr, deltaxcor, l, s

   ! ... Check inputs ...
   derv(1) = 1; derv(2) = 2
   IF (Nr .LT. qdegree) STOP 'N too small for MH scheme'

   infextlowq = 0

   ! ... Uniform grid (input to this subroutine) is overwritten with FD-q when flag > 0 ...
   IF (flag .GT. 0) THEN
      xcor = 0.0d0
      call Optimum_Grid_Nodes(Polynomial_Degree = qdegree, &
                              Nodes             = xcor)
      
!      l = tmax*thalf/(tmax - 2.0d0*thalf)
!      s = 2.d0*l/tmax
   
!      DO i = 1, Nr
!         xcor(i) = l*(1.d0 + xcor(i))/(1.d0 + s - xcor(i))

!      ENDDO
      infextlowq = 0
   ENDIF
  
    
   Polynomial_Degree=qdegree
   call Interpolation_Matrix(Grid_Nodes        = xcor,  &
                          Interpolation_Points = xcor,  &
                          Derivatives          = derv,  &
                          Matrix               = d)

   IF (infextlowq .EQ. 1) THEN
      Polynomial_Degree = qdegree - 1
      CALL Interpolation_Matrix(Grid_Nodes           = xcor, &
                                Interpolation_Points = xcor, &
                                Derivatives          = derv, &
                                Matrix               = d1)


      IF (qdegree .GE. 4) THEN
      Polynomial_Degree = qdegree - 2
      CALL Interpolation_Matrix(Grid_Nodes     = xcor, &
                          Interpolation_Points = xcor, &
                          Derivatives          = derv, &
                          Matrix               = d2)
      ENDIF

      if (qdegree.ge.6) then
         Polynomial_Degree = qdegree - 3
         call Interpolation_Matrix(Grid_Nodes  = xcor, &
                          Interpolation_Points = xcor, &
                          Derivatives          = derv, &
                          Matrix               = d3)
      endif

      if (qdegree.ge.8) then
         Polynomial_Degree = qdegree - 4 
         call Interpolation_Matrix(Grid_Nodes  = xcor, &
                          Interpolation_Points = xcor, &
                          Derivatives          = derv, &
                          Matrix               = d4)
      endif

      if (qdegree.ge.10) then
         Polynomial_Degree=qdegree-5
         call Interpolation_Matrix(Grid_Nodes  = xcor, &
                          Interpolation_Points = xcor, &
                          Derivatives          = derv, &
                          Matrix               = d5)
      endif

      if (qdegree.ge.12) then
         Polynomial_Degree=qdegree-6
         call Interpolation_Matrix(Grid_Nodes  = xcor, &
                          Interpolation_Points = xcor, &
                          Derivatives          = derv, &
                          Matrix               = d6)
      endif

      if (qdegree.eq.2) then
         d (1 ,1,:) = d1(1 ,1,:); d (1 ,2,:) = d1(1 ,2,:)
         d (Nr,1,:) = d1(Nr,1,:); d (Nr,2,:) = d1(Nr,2,:)

      elseif (qdegree.eq.4) then
         d (1 ,1,:)   = d2(1 ,1,:);   d (1 ,2,:)   = d2(1 ,2,:)
         d (Nr,1,:)   = d2(Nr,1,:);   d (Nr,2,:)   = d2(Nr,2,:)

         d (2   ,1,:) = d1(2   ,1,:); d (2   ,2,:) = d1(2   ,2,:)
         d (Nr-1,1,:) = d1(Nr-1,1,:); d (Nr-1,2,:) = d1(Nr-1,2,:)

      elseif (qdegree.eq.6) then
         d (1   ,1,:) = d3(1   ,1,:); d (1   ,2,:) = d3(1   ,2,:)
         d (Nr  ,1,:) = d3(Nr  ,1,:); d (Nr  ,2,:) = d3(Nr  ,2,:)

         d (2   ,1,:) = d2(2   ,1,:); d (2   ,2,:) = d2(2   ,2,:)
         d (Nr-1,1,:) = d2(Nr-1,1,:); d (Nr-1,2,:) = d2(Nr-1,2,:)

         d (3   ,1,:) = d1(3   ,1,:); d (3   ,2,:) = d1(3   ,2,:)
         d (Nr-2,1,:) = d1(Nr-2,1,:); d (Nr-2,2,:) = d1(Nr-2,2,:)

      elseif (qdegree.eq.8) then
         d (1   ,1,:) = d4(1   ,1,:); d (1   ,2,:) = d4(1   ,2,:)
         d (Nr  ,1,:) = d4(Nr  ,1,:); d (Nr  ,2,:) = d4(Nr  ,2,:)

         d (2   ,1,:) = d3(2   ,1,:); d (2   ,2,:) = d3(2   ,2,:)
         d (Nr-1,1,:) = d3(Nr-1,1,:); d (Nr-1,2,:) = d3(Nr-1,2,:)

         d (3   ,1,:) = d2(3   ,1,:); d (3   ,2,:) = d2(3   ,2,:)
         d (Nr-2,1,:) = d2(Nr-2,1,:); d (Nr-2,2,:) = d2(Nr-2,2,:)

         d (4   ,1,:) = d1(4   ,1,:); d (4   ,2,:) = d1(4   ,2,:)
         d (Nr-3,1,:) = d1(Nr-3,1,:); d (Nr-3,2,:) = d1(Nr-3,2,:)

      elseif (qdegree.eq.10) then
         d (1   ,1,:) = d5(1   ,1,:); d (1   ,2,:) = d5(1   ,2,:)
         d (Nr  ,1,:) = d5(Nr  ,1,:); d (Nr  ,2,:) = d5(Nr  ,2,:)

         d (2   ,1,:) = d4(2   ,1,:); d (2   ,2,:) = d4(2   ,2,:)
         d (Nr-1,1,:) = d4(Nr-1,1,:); d (Nr-1,2,:) = d4(Nr-1,2,:)

         d (3   ,1,:) = d3(3   ,1,:); d (3   ,2,:) = d3(3   ,2,:)
         d (Nr-2,1,:) = d3(Nr-2,1,:); d (Nr-2,2,:) = d3(Nr-2,2,:)

         d (4   ,1,:) = d2(4   ,1,:); d (4   ,2,:) = d2(4   ,2,:)
         d (Nr-3,1,:) = d2(Nr-3,1,:); d (Nr-3,2,:) = d2(Nr-3,2,:)

         d (5   ,1,:) = d1(5   ,1,:); d (5   ,2,:) = d1(5   ,2,:)
         d (Nr-4,1,:) = d1(Nr-4,1,:); d (Nr-4,2,:) = d1(Nr-4,2,:)

      elseif (qdegree.eq.12) then
         d (1   ,1,:) = d6(1   ,1,:); d (1   ,2,:) = d6(1   ,2,:)
         d (Nr  ,1,:) = d6(Nr  ,1,:); d (Nr  ,2,:) = d6(Nr  ,2,:)

         d (2   ,1,:) = d5(2   ,1,:); d (2   ,2,:) = d5(2   ,2,:)
         d (Nr-1,1,:) = d5(Nr-1,1,:); d (Nr-1,2,:) = d5(Nr-1,2,:)

         d (3   ,1,:) = d4(3   ,1,:); d (3   ,2,:) = d4(3   ,2,:)
         d (Nr-2,1,:) = d4(Nr-2,1,:); d (Nr-2,2,:) = d4(Nr-2,2,:)

         d (4   ,1,:) = d3(4   ,1,:); d (4   ,2,:) = d3(4   ,2,:)
         d (Nr-3,1,:) = d3(Nr-3,1,:); d (Nr-3,2,:) = d3(Nr-3,2,:)

         d (5   ,1,:) = d2(5   ,1,:); d (5   ,2,:) = d2(5   ,2,:)
         d (Nr-4,1,:) = d2(Nr-4,1,:); d (Nr-4,2,:) = d2(Nr-4,2,:)

         d (6   ,1,:) = d1(6   ,1,:); d (6   ,2,:) = d1(6   ,2,:)
         d (Nr-5,1,:) = d1(Nr-5,1,:); d (Nr-5,2,:) = d1(Nr-5,2,:)

      else
         stop 'qorder>12 for scheme=0 and infextlowq=1?'

      endif
   endif

   DD1 = d(:, 1, :)
   DD2 = d(:, 2, :)
end subroutine STD


SUBROUTINE INTERP(Nr, Nr2, qdegree, xcor, xcor2, DD0)
   USE :: Optimum_Grid_Methods
   USE :: Piecewise_Polynomial_Interpolation

   IMPLICIT NONE
   integer, intent(in) :: Nr, Nr2, qdegree
   real, intent(out) :: DD0(Nr2, Nr)
   real, intent(in) :: xcor(Nr), xcor2(Nr2)

   integer j,i,jcor,icor,ierr,derv(1)
   real d(Nr,1,Nr)
   real res, xLr, deltaxcor, l, s

   ! ... Check inputs ...
   derv(1) = 0;
   IF (Nr .LT. qdegree) STOP 'N too small for MH scheme'
  
    
   Polynomial_Degree=qdegree
   call Interpolation_Matrix(Grid_Nodes        = xcor,  &
                          Interpolation_Points = xcor2,  &
                          Derivatives          = derv,  &
                          Matrix               = d)

   
   DD0 = d(:, 1, :)
end subroutine INTERP
