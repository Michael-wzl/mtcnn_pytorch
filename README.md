# Real MTCNN PyTorch

The primary reason for reimplementing the MTCNN model in PyTorch is that many repos that call themselves "mtcnn_pytorch" do not use PyTorch throught the process, which repeatedly convert tensors back to PIL images before going to the next network. This can cause serious performance issues as data is moved between CPU and GPU. Moreover, many matrix operations like NMS are handwritten and run on CPU, which do not fully utilize the power of GPU and mature libraries like torchvision. These issues haunted me when I tried to find a fast and easy2use version of MTCNN in my latest [paper](). 

Therefore, I rewrote MTCNN in PyTorch completely, with all operations done on GPU and make better use of PyTorch and torchvision. Hope it helps. Plz star the repo if it ever helped you. Thx a lot! (As a CS UG, stars can mean a lot in my resume... BTW, if you are interested in privacy protection against unauthorized face recognition systems, check my latest [paper]() and its [code]())

## Performance Comparison

## Dependencies
