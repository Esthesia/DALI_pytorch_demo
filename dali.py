from typing import MutableMapping
from numpy.matrixlib.defmatrix import matrix
import torch, math
import time

import threading
from torch.multiprocessing import Event
from torch._six import queue

try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator
    from nvidia.dali.plugin.pytorch import DALIGenericIterator
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types
    import nvidia.dali.fn as fn
except ImportError:
    raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")

class RandAugmentPipe(Pipeline):
    print('Using DALI RandAugment iterator')

    def __init__(self, batch_size, num_threads, device_id, data_dir, crop,
                 mean, std, local_rank, world_size, dali_cpu=False, shuffle=True, fp16=False,
                 min_crop_size=0.08, aug_name_list=[], aug_factor=1):
        
        super(RandAugmentPipe, self).__init__(batch_size, num_threads, device_id, seed=-1)
        
        self.input = ops.readers.File(file_root=data_dir, shard_id=local_rank, num_shards=world_size,
                                    random_shuffle=shuffle)
        self.pipe_BatchSize = batch_size
        self.crop = crop
        self.mean = mean
        self.std = std
        if dali_cpu:
            self.decode_device = "cpu"
            self.dali_device = "cpu"
            output_dtype = types.FLOAT
            self.flip = ops.Flip(device=self.dali_device)
        
            self.augmentations = {}
            self.meta_augmentations = []
            for op, minval, maxval in aug_name_list:
                self.meta_augmentations.append(op)
                if op == "Brightness":
                    val = (float(aug_factor) / 30) * float(maxval - minval) + minval
                    '''
                    fn.brightness_contrast
                    
                    out = brightness_shift * output_range + brightness \
                        * (contrast_center + contrast * (in - contrast_center))
                    '''
                    self.augmentations["Brightness"] = \
                        lambda images: fn.brightness_contrast(images,
                                                              brightness_shift=val)
                if op == "Contrast":
                    val = (float(aug_factor) / 30) * float(maxval - minval) + minval
                    self.augmentations["Contrast"] = \
                        lambda images: fn.brightness_contrast(images,
                                                              contrast=val)

                if op == "Rotate":
                    val = (float(aug_factor) / 30) * float(maxval - minval) + minval                    
                    self.augmentations["Rotate"] = \
                        lambda images: fn.rotate(images,
                                                 angle=val,
                                                 interp_type=types.INTERP_LINEAR,
                                                 fill_value=0)
                if op == "Invert":
                    val = (float(aug_factor) / 30) * float(maxval - minval) + minval
                    # Color value inverting - implement with flip for convenience
                    self.augmentations["Invert"] = \
                        lambda images: fn.flip(images,
                                            vertical=0,
                                            horizontal=1)
                    
                if op == "ShearX":
                    val = (float(aug_factor) / 30) * float(maxval - minval) + minval 
                    # ShearX img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))
                    self.augmentations["ShearX"] = \
                        lambda images: fn.warp_affine(images,
                                                      matrix=[1.0, val, 0.0, 0.0, 1.0, 0.0],
                                                      interp_type=types.INTERP_LINEAR)
                if op == "ShearY":
                    val = (float(aug_factor) / 30) * float(maxval - minval) + minval
                    # ShearY img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))
                    self.augmentations["ShearY"] = \
                        lambda images: fn.warp_affine(images,
                                                      matrix=[1.0, 0.0, 0.0, val, 1.0, 0.0],
                                                      interp_type=types.INTERP_LINEAR)
                if op == "TranslateXabs":
                    val = (float(aug_factor) / 30) * float(maxval - minval) + minval    
                    # TranslateX abs img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0)) 
                    self.augmentations["TranslateXabs"] = \
                        lambda images: fn.warp_affine(images,
                                                      matrix=[1.0, 0.0, val, 0.0, 1.0, 0.0],
                                                      interp_type=types.INTERP_LINEAR)
                if op == "TranslateYabs":
                    val = (float(aug_factor) / 30) * float(maxval - minval) + minval
                    # TranslateY abs img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))
                    self.augmentations["TranslateYabs"] = \
                        lambda images: fn.warp_affine(images,
                                                      matrix=[1.0, 0.0, 0.0, 0.0, 1.0, val],
                                                      interp_type=types.INTERP_LINEAR)


        else:
            self.decode_device = "mixed"
            self.dali_device = "gpu"

            output_dtype = types.FLOAT
            if self.dali_device == "gpu" and fp16:
                output_dtype = types.FLOAT16            

            # self.cmn = ops.CropMirrorNormalize(device="gpu",
            #                                    dtype=output_dtype,
            #                                    output_layout=types.NCHW,
            #                                    crop=(crop, crop),
            #                                    mean=mean,
            #                                    std=std,)

            self.augmentations = {}
            self.meta_augmentations = []          
            for op, minval, maxval in aug_name_list:
                self.meta_augmentations.append(op)
                if op == "Brightness":
                    val = (float(aug_factor) / 30) * float(maxval - minval) + minval
                    '''
                    fn.brightness_contrast
                    
                    out = brightness_shift * output_range + brightness \
                        * (contrast_center + contrast * (in - contrast_center))
                    '''
                    self.augmentations["Brightness"] = \
                        lambda images: fn.brightness_contrast(images,
                                                            brightness_shift=val)
                if op == "Contrast":
                    val = (float(aug_factor) / 30) * float(maxval - minval) + minval
                    self.augmentations["Contrast"] = \
                        lambda images: fn.brightness_contrast(images,
                                                            contrast=val)

                if op == "Rotate":
                    val = (float(aug_factor) / 30) * float(maxval - minval) + minval                    
                    self.augmentations["Rotate"] = \
                        lambda images: fn.rotate(images,
                                                angle=val,
                                                interp_type=types.INTERP_LINEAR,
                                                fill_value=0)
                if op == "Invert":
                    val = (float(aug_factor) / 30) * float(maxval - minval) + minval
                    # Color value inverting - implement with flip for convenience
                    self.augmentations["Invert"] = \
                        lambda images: fn.flip(images,
                                            vertical=0,
                                            horizontal=1)
                    
                if op == "ShearX":
                    val = (float(aug_factor) / 30) * float(maxval - minval) + minval 
                    # ShearX img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))
                    self.augmentations["ShearX"] = \
                        lambda images: fn.warp_affine(images,
                                                    matrix=[1.0, val, 0.0, 0.0, 1.0, 0.0],
                                                    interp_type=types.INTERP_LINEAR)
                if op == "ShearY":
                    val = (float(aug_factor) / 30) * float(maxval - minval) + minval
                    # ShearY img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))
                    self.augmentations["ShearY"] = \
                        lambda images: fn.warp_affine(images,
                                                    matrix=[1.0, 0.0, 0.0, val, 1.0, 0.0],
                                                    interp_type=types.INTERP_LINEAR)
                if op == "TranslateXabs":
                    val = (float(aug_factor) / 30) * float(maxval - minval) + minval    
                    # TranslateX abs img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0)) 
                    self.augmentations["TranslateXabs"] = \
                        lambda images: fn.warp_affine(images,
                                                    matrix=[1.0, 0.0, val, 0.0, 1.0, 0.0],
                                                    interp_type=types.INTERP_LINEAR)
                if op == "TranslateYabs":
                    val = (float(aug_factor) / 30) * float(maxval - minval) + minval
                    # TranslateY abs img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))
                    self.augmentations["TranslateYabs"] = \
                        lambda images: fn.warp_affine(images,
                                                    matrix=[1.0, 0.0, 0.0, 0.0, 1.0, val],
                                                    interp_type=types.INTERP_LINEAR)

        self.coin = ops.random.CoinFlip(probability=0.5)
        # print('DALI "{0}" variant'.format(self.dali_device))                                        

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name="Reader")            

        # Combined decode & random crop
        images = fn.decoders.image(self.jpegs, device=self.decode_device)
        # Resize as desired
        images = fn.resize(images, resize_x = self.crop, resize_y=self.crop)
           
        
        # print(self.pipe_BatchSize)

        if self.dali_device == "gpu":
            print(self.meta_augmentations)
            for aug in self.augmentations.values():
                images = aug(images)
            output = fn.crop_mirror_normalize(images,
                                              mirror = rng,
                                              crop=[self.crop, self.crop], 
                                              mean = self.mean,
                                              std = self.std,
                                              dtype=types.FLOAT,
                                              output_layout=types.NCHW)
        else:
            print("Working on CPU side")
            print(self.meta_augmentations)
            for aug in self.augmentations.values():
                images = aug(images)
            output = fn.crop_mirror_normalize(images,
                                              mirror = rng,
                                              crop=[self.crop, self.crop], 
                                              mean = self.mean,
                                              std = self.std,
                                              dtype=types.FLOAT,
                                              output_layout=types.NCHW)
            output = self.flip(images, horizontal=rng)
        self.labels = self.labels.gpu()
        return [output, self.labels]        
class DaliIterator():
    """
    Wrapper class to decode the DALI iterator output & provide iterator that functions the same as torchvision

    pipelines (Pipeline): DALI pipelines
    size (int): Number of examples in set

    Note: allow extra inputs to keep compatibility with CPU iterator
    """

    def __init__(self, pipelines, size, **kwargs):
        self._dali_iterator = DALIClassificationIterator(pipelines=pipelines, size=size)

    def __iter__(self):
        return self

    def __len__(self):
        return int(math.ceil(self._dali_iterator._size / self._dali_iterator.batch_size))


class DaliIteratorGPU(DaliIterator):
    """
    Wrapper class to decode the DALI iterator output & provide iterator that functions the same as torchvision

    pipelines (Pipeline): DALI pipelines
    size (int): Number of examples in set

    Note: allow extra inputs to keep compatibility with CPU iterator
    """

    def __next__(self):
        try:
            data = next(self._dali_iterator)
        except StopIteration:
            print('Resetting DALI loader')
            self._dali_iterator.reset()
            raise StopIteration

        # Decode the data output
        input = data[0]['data']
        target = data[0]['label'].squeeze().long()

        return input, target

        
def _preproc_worker(dali_iterator, cuda_stream, fp16, mean, std, output_queue, proc_next_input, done_event, pin_memory):
    """
    Worker function to parse DALI output & apply final pre-processing steps
    """

    while not done_event.is_set():
        # Wait until main thread signals to proc_next_input -- normally once it has taken the last processed input
        proc_next_input.wait()
        proc_next_input.clear()

        if done_event.is_set():
            # print('Shutting down preproc thread')
            break

        try:
            data = next(dali_iterator)

            # Decode the data output
            input_orig = data[0]['data']
            target = data[0]['label'].squeeze().long()  # DALI should already output target on device

            # Copy to GPU and apply final processing in separate CUDA stream
            # with torch.cuda.stream(cuda_stream):
            #     input = input_orig
            #     if pin_memory:
            #         input = input.pin_memory()
            #         del input_orig  # Save memory
            #     input = input.cuda(non_blocking=True)

            #     input = input.permute(0, 3, 1, 2)

            #     # Input tensor is kept as 8-bit integer for transfer to GPU, to save bandwidth
            #     if fp16:
            #         input = input.half()
            #     else:
            #         input = input.float()

            #     input = input.sub_(mean).div_(std)

            # Put the result on the queue
            output_queue.put((input, target))

        except StopIteration:
            print('Resetting DALI loader')
            dali_iterator.reset()
            output_queue.put(None)


class DaliIteratorCPU(DaliIterator):
    """
    Wrapper class to decode the DALI iterator output & provide iterator that functions the same as torchvision
    Note that permutation to channels first, converting from 8 bit to float & normalization are all performed on GPU

    pipelines (Pipeline): DALI pipelines
    size (int): Number of examples in set
    fp16 (bool): Use fp16 as output format, f32 otherwise
    mean (tuple): Image mean value for each channel
    std (tuple): Image standard deviation value for each channel
    pin_memory (bool): Transfer input tensor to pinned memory, before moving to GPU
    """
    def __init__(self, fp16=False, mean=(0., 0., 0.), std=(1., 1., 1.), pin_memory=True, **kwargs):
        super().__init__(**kwargs)
        print('Using DALI CPU iterator')
        self.stream = torch.cuda.Stream()

        self.fp16 = fp16
        self.mean = torch.tensor(mean).cuda().view(1, 3, 1, 1)
        self.std = torch.tensor(std).cuda().view(1, 3, 1, 1)
        self.pin_memory = pin_memory

        if self.fp16:
            self.mean = self.mean.half()
            self.std = self.std.half()

        self.proc_next_input = Event()
        self.done_event = Event()
        self.output_queue = queue.Queue(maxsize=5)
        self.preproc_thread = threading.Thread(
            target=_preproc_worker,
            kwargs={'dali_iterator': self._dali_iterator, 'cuda_stream': self.stream, 'fp16': self.fp16, 'mean': self.mean, 'std': self.std, 'proc_next_input': self.proc_next_input, 'done_event': self.done_event, 'output_queue': self.output_queue, 'pin_memory': self.pin_memory})
        self.preproc_thread.daemon = True
        self.preproc_thread.start()

        self.proc_next_input.set()

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        data = self.output_queue.get()
        self.proc_next_input.set()
        if data is None:
            raise StopIteration
        return data

    def __del__(self):
        self.done_event.set()
        self.proc_next_input.set()
        torch.cuda.current_stream().wait_stream(self.stream)
        self.preproc_thread.join()


class DaliIteratorCPUNoPrefetch(DaliIterator):
    """
    Wrapper class to decode the DALI iterator output & provide iterator that functions the same as torchvision
    Note that permutation to channels first, converting from 8 bit to float & normalization are all performed on GPU

    pipelines (Pipeline): DALI pipelines
    size (int): Number of examples in set
    fp16 (bool): Use fp16 as output format, f32 otherwise
    mean (tuple): Image mean value for each channel
    std (tuple): Image standard deviation value for each channel
    pin_memory (bool): Transfer input tensor to pinned memory, before moving to GPU
    """
    def __init__(self, fp16, mean, std, pin_memory=True, **kwargs):
        super().__init__(**kwargs)
        print('Using DALI CPU iterator')

        self.stream = torch.cuda.Stream()

        self.fp16 = fp16
        self.mean = torch.tensor(mean).cuda().view(1, 3, 1, 1)
        self.std = torch.tensor(std).cuda().view(1, 3, 1, 1)
        self.pin_memory = pin_memory

        if self.fp16:
            self.mean = self.mean.half()
            self.std = self.std.half()

    def __next__(self):
        data = next(self._dali_iterator)

        # Decode the data output
        input = data[0]['data']
        target = data[0]['label'].squeeze().long()  # DALI should already output target on device

        # Copy to GPU & apply final processing in seperate CUDA stream
        input = input.cuda(non_blocking=True)

        input = input.permute(0, 3, 1, 2)

        # Input tensor is transferred to GPU as 8 bit, to save bandwidth
        if self.fp16:
            input = input.half()
        else:
            input = input.float()

        input = input.sub_(self.mean).div_(self.std)
        return input, target
