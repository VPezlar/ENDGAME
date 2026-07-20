program main
  implicit none
  integer :: i
  integer :: n, q
  real, allocatable, dimension(:) :: eta
  real, allocatable, dimension(:, :) :: d1, d2

  open(1,file='../FDq/inputs/input_size.dat')
     read(1,*) n
     read(1,*) q
  close(1)

  allocate(eta(n),d1(n,n),d2(n,n))
  i=1
  do i=1,n
    eta(i) = -1.0d0 + (2.0d0/(n-1))*(i-1)  
  end do
  
  call STD(n, q, eta, d1, d2, 1)
  
  open(unit = 1, file = '../FDq/output/d1.dat')
  do i = 1, n
    write(1, *) d1(i, :)
  enddo
  close(1)

  open(unit = 2, file = '../FDq/output/d2.dat')
  do i = 1, n
    write(2, *) d2(i, :)
  enddo
  close(2)

  open(unit = 3, file = '../FDq/output/eta.dat')
  do i = 1, n
    !write(3, '(11(E16.10,3X))') eta(i)
    write(3, *) eta(i)
  enddo
  close(3)

deallocate(eta,d1,d2)

end program main
