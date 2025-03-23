use mnist::*;
use ndarray::prelude::*;

pub fn mnist_loader( print: bool) -> (Array3<f32>, Array2<f32>, Array3<f32>, Array2<f32>) {

   // Deconstruct the returned Mnist struct.
   let Mnist {
       trn_img,
       trn_lbl,
       tst_img,
       tst_lbl,
       ..
   } = MnistBuilder::new()
       .label_format_digit()
       .training_set_length(50_000)
       .validation_set_length(10_000)
       .test_set_length(10_000)
       .finalize();

   let image_num = 0;
   
   // Can use an Array2 or Array3 here (Array3 for visualization)
   let train_data = Array3::from_shape_vec((50_000, 28, 28), trn_img)
       .expect("Error converting images to Array3 struct")
       .map(|x| *x as f32 / 256.0);
    if print {
        println!("{:#.1?}",train_data.slice(s![image_num, .., ..]));
    }

   // Convert the returned Mnist struct to Array2 format
   let train_labels: Array2<f32> = Array2::from_shape_vec((50_000, 1), trn_lbl)
       .expect("Error converting training labels to Array2 struct")
       .map(|x| *x as f32);
    if print { 
        println!("The first digit is a {:?}",train_labels.slice(s![image_num, ..]) );
    }

   let test_data = Array3::from_shape_vec((10_000, 28, 28), tst_img)
       .expect("Error converting images to Array3 struct")
       .map(|x| *x as f32 / 256.);

   let test_labels: Array2<f32> = Array2::from_shape_vec((10_000, 1), tst_lbl)
       .expect("Error converting testing labels to Array2 struct")
       .map(|x| *x as f32);

       (train_data, train_labels, test_data, test_labels)
   
}


