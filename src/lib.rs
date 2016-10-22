extern crate ndarray;
extern crate rand;
extern crate ndarray_rand;
extern crate rayon;

pub mod ops;

#[cfg(test)]
mod tests {
    use ops::Matrix;
    use ndarray::{Array, arr2};
    use ndarray_rand::RandomExt;
    use rand::distributions::Range;
    #[test]
    fn test_dot() {
        let x = Matrix { inner: Array::random((50, 50), Range::new(0., 10.)) };

        let y = Matrix { inner: Array::random((50, 50), Range::new(0., 10.)) };
        let res = x.clone() * y.clone();
        let res1 = x.inner.dot(&y.inner);

        assert!(res.inner.all_close(&res1, 0.01))

    }
    #[test]
    fn test_add() {
        let a = Matrix { inner: Array::random((50, 50), Range::new(0., 10.)) };

        let b = Matrix { inner: Array::random((50, 50), Range::new(0., 10.)) };
        let res = a.clone() + b.clone();

        let mut res1 = a.inner.to_owned();
        res1.view_mut().zip_mut_with(&b.inner, |x, y| *x = *x + *y);

        assert!(res.inner.all_close(&res1, 0.01))

    }

}
