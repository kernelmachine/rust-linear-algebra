use std::ops::{Mul, Add};
use rayon;
use ndarray::{Array, ArrayView, ArrayViewMut, Ix, Axis};


#[derive(Debug, Clone)]
pub struct Matrix<f64> {
    pub inner: Array<f64, (Ix, Ix)>,
}



pub const BLOCKSIZE: usize = 50;

impl Add for Matrix<f64> {
    type Output = Matrix<f64>;
    fn add(self, _rhs: Matrix<f64>) -> Matrix<f64> {
        fn matrix_add_safe(left: ArrayView<f64, (Ix, Ix)>,
                           right: ArrayView<f64, (Ix, Ix)>,
                           init: &mut ArrayViewMut<f64, (Ix, Ix)>) {
            let res = left.to_owned() + right;
            init.zip_mut_with(&res, |x, y| *x = *y);
            return;
        }

        fn matrix_add_rayon(left: ArrayView<f64, (Ix, Ix)>,
                            right: ArrayView<f64, (Ix, Ix)>,
                            init: &mut ArrayViewMut<f64, (Ix, Ix)>) {

            let (m, _) = left.dim();
            let (_, n) = right.dim();

            if m <= BLOCKSIZE && n <= BLOCKSIZE {
                matrix_add_safe(left, right, init);
            } else if m > BLOCKSIZE {
                let mid = m / 2;
                let (left_0, left_1) = left.view().split_at(Axis(0), mid);
                let (mut init_left, mut init_right) = init.view_mut().split_at(Axis(0), mid);
                rayon::join(|| matrix_add_rayon(left_1, right, &mut init_left),
                            || matrix_add_rayon(left_0, right, &mut init_right));

            } else if n > BLOCKSIZE {
                let mid = n / 2;

                let (right_0, right_1) = right.view().split_at(Axis(1), mid);
                let (mut init_left, mut init_right) = init.view_mut().split_at(Axis(1), mid);
                rayon::join(|| matrix_add_rayon(left, right_0, &mut init_left),
                            || matrix_add_rayon(left, right_1, &mut init_right));
            }

        }
        let (m, k1) = self.inner.dim();
        let (k2, n) = _rhs.inner.dim();

        assert_eq!(k1, k2);
        assert_eq!(m, n);

        let mut init = Array::zeros((m, n));

        matrix_add_rayon(self.inner.view(), _rhs.inner.view(), &mut init.view_mut());

        Matrix { inner: init }
    }
}
impl Mul for Matrix<f64> {
    type Output = Matrix<f64>;

    fn mul(self, _rhs: Matrix<f64>) -> Matrix<f64> {


        fn matrix_dot_safe(left: ArrayView<f64, (Ix, Ix)>,
                           right: ArrayView<f64, (Ix, Ix)>,
                           init: &mut ArrayViewMut<f64, (Ix, Ix)>) {
            let res = left.dot(&right);

            init.zip_mut_with(&res, |x, y| *x = *y);

            return;
        }

        fn matrix_dot_rayon(left: ArrayView<f64, (Ix, Ix)>,
                            right: ArrayView<f64, (Ix, Ix)>,
                            init: &mut ArrayViewMut<f64, (Ix, Ix)>) {

            let (m, _) = left.dim();
            let (_, n) = right.dim();

            if m <= BLOCKSIZE && n <= BLOCKSIZE {
                matrix_dot_safe(left, right, init);
            } else if m > BLOCKSIZE {
                let mid = m / 2;
                let (left_0, left_1) = left.view().split_at(Axis(0), mid);
                let (mut init_left, mut init_right) = init.view_mut().split_at(Axis(0), mid);
                rayon::join(|| matrix_dot_rayon(left_1, right, &mut init_left),
                            || matrix_dot_rayon(left_0, right, &mut init_right));

            } else if n > BLOCKSIZE {
                let mid = n / 2;

                let (right_0, right_1) = right.view().split_at(Axis(1), mid);
                let (mut init_left, mut init_right) = init.view_mut().split_at(Axis(1), mid);
                rayon::join(|| matrix_dot_rayon(left, right_0, &mut init_left),
                            || matrix_dot_rayon(left, right_1, &mut init_right));
            }



        }

        let (_, k1) = self.inner.dim();
        let (k2, _) = _rhs.inner.dim();

        assert_eq!(k1, k2);

        let mut init = Array::zeros((k1, k2));

        matrix_dot_rayon(self.inner.view(), _rhs.inner.view(), &mut init.view_mut());

        Matrix { inner: init }

    }
}
