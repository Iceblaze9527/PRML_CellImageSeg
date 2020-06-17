# PRML_PRO_CellImageSeg

Course Project of 2020-S Pattern Recognition and Machine Learning, instructed by Wang Xiaowo and Zhang Xuegong in Tsinghua University.

The project aims to achieve two cell segmentation tasks using U-Net implemented by PyTorch. 

Credits:
* U-Net implementation: [@jvanvugt](https://github.com/jvanvugt/pytorch-unet)
* Focal Loss implementation: [@clcarwin](https://github.com/clcarwin/focal_loss_pytorch)
* Dice Loss implementation: [missing, find it later]()
* Weighted Loss implementation: [@jaidevd](https://jaidevd.github.io/posts/weighted-loss-functions-for-instance-segmentation/)

`PRML_Report.md` is the project report (in Chinese).

## Project Structure
* `_credit/`: all credited codes and licenses
  * `README_unet.md`: readme file for unet implementation, the author addressed some important issues with respect to U-Net architecture, good to read.
  * `focalloss.py`
  * `unet.py`
  * `LICENSE_focal_loss`
  * `LICENSE_unet`
* `_demo/`: data conversion/visualization/output demo
  * `dataset1_demo.py`: inout demo on dataset1
  * `visual_demo.py`: visualization demo
  * `image_io.ipynb`: jupyter notebook version of `dataset1_demo.py`
  * `image_vis.ipynb`: jupyter notebook version of `visual_demo.py`
* `dataset1`: task1 dataset
  * `test/`: test dataset, uint16 single channel images
  * `test_RES/`: sample test predictions, uint16 single channel images
  * `train/`: train dataset, uint16 single channel images
  * `train_GT/SEG`: ground truth labels, uint16 single channel images
  * `visual`: visualization of original images and its labels
* `dataset2`: task2 dataset
  * `test/`: test dataset, uint16 single channel images
  * `train/`: train dataset, uint16 single channel images (not all are annotated)
  * `train_GT`: ground truth labels, uint16 single channel images
    * `SEG/`: given annotated images
    * `SEG_part/`: add `labelme` annotated results, 29 records in total
  * `train_part/`: original images corresponding to `SEG_part/`
  * `visual`: visualization of original images and its labels
* `labelme/`: `labelme` annotated results
  * `done_json/`: `labelme` json outputs
  * `done_png/`: png labels converted from `done_json/`
  * `script.py`: convert png labels to ground truth formats, i.e. uint16 single channel images
* `deprecated.py`：deprecated codes
	* `weighted_map()`: weighted loss function
	* `post_processing()`: OpenCV implementation of watershed algorithm, see [here](https://docs.opencv.org/master/d3/db4/tutorial_py_watershed.html)
* `DG.py`：Dataset generation class module
* `Loss.py`：self-defined loss functions
	* `focal_loss()`
	* `dice_loss()`
* `Oper.py`：model operation
	* `run_model()`
	* `train_model()`
	* `val_model()`
* `Output.py`：output module
	* `test_out()`：output predictions on test dataset
	* `post_processing()`：segment pixels to different instances
* `Stats.py`：data analysis module
	* `print_data()`：print loss and binary jaccard
* `UNet.py`：U-Net implementation
* `continue.ipynb`：continue training model checkpoint
* `output.ipynb`：output predictions on test dataset
* `task1.ipynb`：task1 main
* `task2.ipynb`：task2 main
* `config.yaml`：`anaconda` environment configuration
