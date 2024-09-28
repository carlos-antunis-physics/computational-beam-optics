!------------------------------------------------------------------------------
!
!   @author
!>      Carlos Antunis Bonfim da Silva Santos
!>      https://github.com/carlos-antunis-physics 
!
!   DESCRIPTION: 
!>      Utils subroutines on computational approach of optics research.
!
!>  f2py compiling:
!>      f2py -m linear_algebra linear_algebra.f95 -h linear_algebra.pyf
!>      f2py -c linear_algebra.pyf linear_algebra.f95
!
!------------------------------------------------------------------------------
module linear_algebra
contains
    function thomas(n, lower_diagonal, diagonal, upper_diagonal, b) result(x)
        !
        !   optical.Propagation.linear_algebra.linear_algebra.thomas
        !       solves the linear system of equations Tx = b in which T is a
        !       tridiagonal matrix.
        !
        integer, intent(in) :: n;               ! number of equations to solve
        ! equation system coefficients
        double complex, dimension(n), intent(in) :: diagonal;
        double complex, dimension(n - 1), intent(in) :: lower_diagonal, upper_diagonal;
        double complex, dimension(n), intent(in) :: b;
        ! solution of equation system
        double complex, dimension(n) :: x;
        ! local auxiliary variables
        double complex, dimension(n) :: y;
        double complex :: m;
        ! initialize variables
        y(1) = upper_diagonal(1) / diagonal(1);
        x(1) = b(1) / diagonal(1);
        ! forward elimination
        do i_row = 2, n
            m = diagonal(i_row) - y(i_row - 1) * lower_diagonal(i_row);
            y(i_row) = upper_diagonal(i_row) / m;
            x(i_row) = (b(i_row) - x(i_row - 1) * lower_diagonal(i_row)) / m;
        end do
        ! backward substitution
        do i_row = n - 1, 1, -1
            x(i_row) = x(i_row) - y(i_row) * x(i_row + 1);
        end do
    end function thomas
end module linear_algebra