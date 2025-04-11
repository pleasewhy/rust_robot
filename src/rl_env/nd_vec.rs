use burn::tensor::backend::Backend;
use burn::tensor::Bool;
use burn::tensor::Element;
use burn::tensor::Tensor;
use burn::tensor::TensorData;
use std::ops::{Index, IndexMut};
use std::usize;

use num_traits::{ToPrimitive, Zero};

#[derive(Debug)]
pub struct NdVec<T: Clone + Zero + ToPrimitive, const D: usize> {
    raw_vec: Vec<T>,
    shape: [usize; D],
}

impl<T: Clone + Zero + ToPrimitive, const D: usize> Clone for NdVec<T, D> {
    fn clone(&self) -> Self {
        Self {
            raw_vec: self.raw_vec.clone(),
            shape: self.shape.clone(),
        }
    }
}
impl<T: Clone + Zero + ToPrimitive, const D: usize> NdVec<T, D> {
    pub fn shape(&self) -> [usize; D] {
        return self.shape;
    }
    pub fn into_vec(self) -> Vec<T> {
        return self.raw_vec;
    }
    pub fn as_slice(&self) -> &[T] {
        self.raw_vec.as_slice()
    }
    pub fn from_shape_vec(vec: Vec<T>, shape: [usize; D]) -> Self {
        Self {
            raw_vec: vec,
            shape: shape,
        }
    }
    pub fn num_elements(&self) -> usize {
        if self.shape.is_empty() {
            return 0;
        }
        let mut sz = self.shape[0];
        for x in &self.shape.as_slice()[1..] {
            sz *= x;
        }
        return sz;
    }
    pub fn to_f64(&self) -> NdVec<f64, D> {
        let vec = self
            .raw_vec
            .iter()
            .map(|x| x.to_f64().unwrap())
            .collect::<Vec<f64>>();
        return NdVec {
            raw_vec: vec,
            shape: self.shape,
        };
    }
}

impl<T: Clone + Zero + ToPrimitive> NdVec<T, 2> {
    // 计算二维索引位置
    fn calculate_index(&self, (row, col): (usize, usize)) -> usize {
        row * self.shape[1] + col
    }

    pub fn zeros(shape: (usize, usize)) -> Self {
        let len = shape.0 * shape.1;
        let raw_vec = vec![T::zero(); len];
        let shape = [shape.0, shape.1];
        return Self { raw_vec, shape };
    }
    pub fn row(&self, row_id: usize) -> Option<&[T]> {
        let start = row_id * self.shape[1];
        let end = start + self.shape[1];
        if end <= self.raw_vec.len() {
            return Some(&self.raw_vec[start..end]);
        }
        return None;
    }
    pub fn row_mut(&mut self, row_id: usize) -> Option<&mut [T]> {
        let start = row_id * self.shape[1];
        let end = start + self.shape[1];
        if end <= self.raw_vec.len() {
            return Some(&mut self.raw_vec[start..end]);
        }
        return None;
    }
    pub fn get(&self, shape: (usize, usize)) -> Option<&T> {
        return self.raw_vec.get(self.calculate_index(shape));
    }
}

// 为NdVec2实现Index和IndexMut trait
impl<T: Clone + Zero + ToPrimitive> Index<(usize, usize)> for NdVec<T, 2> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.raw_vec[self.calculate_index(index)]
    }
}

impl<T: Clone + Zero + ToPrimitive> IndexMut<(usize, usize)> for NdVec<T, 2> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let idx = self.calculate_index(index);
        &mut self.raw_vec[idx]
    }
}

impl<T: Clone + Zero + ToPrimitive> NdVec<T, 3> {
    // 计算三维索引位置
    fn calculate_index(&self, (d0, d1, d2): (usize, usize, usize)) -> usize {
        d0 * self.shape[1] * self.shape[2] + d1 * self.shape[2] + d2
    }

    pub fn zeros(shape: (usize, usize, usize)) -> Self {
        let len = shape.0 * shape.1 * shape.2;
        let raw_vec = vec![T::zero(); len];
        let shape = [shape.0, shape.1, shape.2];
        return Self { raw_vec, shape };
    }

    pub fn slice_mut(&mut self, shape: (usize, usize)) -> Option<&mut [T]> {
        let dim0_id = shape.0;
        let dim1_id = shape.1;
        let start = dim0_id * self.shape[1] * self.shape[2] + dim1_id * self.shape[2];
        let end = start + self.shape[2];
        if end <= self.raw_vec.len() {
            return Some(&mut self.raw_vec[start..end]);
        }
        return None;
    }
    pub fn get(&self, shape: (usize, usize, usize)) -> Option<&T> {
        return self.raw_vec.get(self.calculate_index(shape));
    }
    pub fn as_nd_array(&self) -> ndarray::ArrayView3<T> {
        ndarray::ArrayView3::from_shape(self.shape, self.raw_vec.as_slice()).unwrap()
    }
}

// 为NdVec3实现Index和IndexMut trait
impl<T: Clone + Zero + ToPrimitive> Index<(usize, usize, usize)> for NdVec<T, 3> {
    type Output = T;

    fn index(&self, index: (usize, usize, usize)) -> &Self::Output {
        &self.raw_vec[self.calculate_index(index)]
    }
}

impl<T: Clone + Zero + ToPrimitive> IndexMut<(usize, usize, usize)> for NdVec<T, 3> {
    fn index_mut(&mut self, index: (usize, usize, usize)) -> &mut Self::Output {
        let idx = self.calculate_index(index);
        &mut self.raw_vec[idx]
    }
}

pub type NdVec2<A> = NdVec<A, 2>;
pub type NdVec3<A> = NdVec<A, 3>;

pub fn vec2tensor1<B: Backend, T: Element + Zero + ToPrimitive>(
    arr: Vec<T>,
    device: &B::Device,
) -> Tensor<B, 1> {
    let shape = [arr.len()];
    let tensor_data = TensorData::new(arr, shape);
    return Tensor::<B, 1>::from_data(tensor_data, device);
}

pub fn vec2tensor2<B: Backend, T: Element + Zero + ToPrimitive>(
    arr: NdVec2<T>,
    device: &B::Device,
) -> Tensor<B, 2> {
    let shape = arr.shape();
    let tensor_data = TensorData::new(arr.into_vec(), shape);
    return Tensor::<B, 2>::from_data(tensor_data, device);
}

pub fn vec2tensor3<B: Backend, T: Element + Zero + ToPrimitive>(
    arr: NdVec3<T>,
    device: &B::Device,
) -> Tensor<B, 3> {
    let shape: [usize; 3] = arr.shape();
    let tensor_data = TensorData::new(arr.into_vec(), shape);
    return Tensor::<B, 3>::from_data(tensor_data, device);
}

pub fn tensor2vec2<B: Backend>(tensor: &Tensor<B, 2>) -> NdVec2<f32> {
    let vec = tensor.to_data().into_vec::<f32>().unwrap();
    return NdVec2::from_shape_vec(vec, tensor.shape().dims());
}
pub fn tensor2vec1<B: Backend>(tensor: &Tensor<B, 1>) -> Vec<f32> {
    let vec = tensor.to_data().into_vec::<f32>().unwrap();
    return vec;
}
pub fn booltensor2vec1<B: Backend>(tensor: &Tensor<B, 1, Bool>) -> Vec<bool> {
    let vec = tensor.to_data().into_vec::<bool>().unwrap();
    return vec;
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::Zero;

    #[test]
    fn test_ndvec2_zeros() {
        let v = NdVec2::<i32>::zeros((3, 4));
        assert_eq!(v.shape, [3, 4]);
        assert!(v.raw_vec.iter().all(|&x| x == 0));
    }

    #[test]
    fn test_row_mut() {
        let mut v = NdVec2::zeros((2, 2));
        if let Some(row) = v.row_mut(0) {
            row[1] = 5;
        }
        assert_eq!(v.row(0), Some(&[0, 5][..]));
    }

    #[test]
    fn test_ndvec3_slice() {
        let mut v = NdVec3::zeros((2, 3, 4));
        if let Some(slice) = v.slice_mut((1, 2)) {
            slice[0] = 5;
        }
        assert_eq!(v.raw_vec[1 * 3 * 4 + 2 * 4 + 0], 5);
    }

    #[test]
    fn test_index_2d() {
        let mut v = NdVec2::zeros((2, 3));
        v[(0, 1)] = 5;
        v[(1, 2)] = 7;

        assert_eq!(v[(0, 1)], 5);
        assert_eq!(v[(1, 2)], 7);
        assert_eq!(v.raw_vec, vec![0, 5, 0, 0, 0, 7]);
    }

    #[test]
    #[should_panic]
    fn test_index_out_of_range() {
        let v = NdVec2::<i32>::zeros((1, 1));
        let _ = v[(1, 0)];
    }

    #[test]
    fn test_index_3d() {
        let mut v = NdVec3::zeros((2, 3, 4));
        v[(1, 2, 0)] = 5;
        assert_eq!(v.raw_vec[1 * 3 * 4 + 2 * 4 + 0], 5);
    }

    #[test]
    fn test_invalid_access() {
        let v = NdVec2::<f64>::zeros((1, 1));
        assert_eq!(v.row(1), None);

        let mut v3 = NdVec3::<f64>::zeros((1, 1, 1));
        assert_eq!(v3.slice_mut((1, 0)), None);
    }
}
