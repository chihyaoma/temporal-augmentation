--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
local M = { }

function M.parse(arg)
   local cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Torch-7 Training script')
   cmd:text()
   cmd:text('Options:')
    ------------ General options --------------------
   cmd:option('-rgbData',        '/media/chih-yao/ssd-data/ucf101/features/frameLevelFeatures/rgb/',         'Path to dataset')
   cmd:option('-flowData',       '/media/chih-yao/ssd-data/ucf101/features/frameLevelFeatures/flow/',         'Path to dataset')
   cmd:option('-dataset',        'ucf101',       'Options: ucf101 | hmdb51 ')
   cmd:option('-flowDataset',    'ucf101-flow',  'Options: ucf101-flow | hmdb-flow')
   cmd:option('-split',          1,              'Options: 1 | 2 | 3')
   cmd:option('-manualSeed',     0,              'Manually set RNG seed')
   cmd:option('-nGPU',           1,              'Number of GPUs to use by default')
   cmd:option('-useDevice',      1,              'which GPU to use')
   cmd:option('-backend',        'cudnn',        'Options: cudnn | cunn')
   cmd:option('-cudnn',          'fastest',      'Options: fastest | default | deterministic')
   cmd:option('-gen',            'gen',          'Path to save generated files')
   cmd:option('-precision',      'single',       'Options: single | double | half')
   ------------- Data options ------------------------
   cmd:option('-nThreads',            4,        'number of data loading threads')
   cmd:option('-nVideoSeg',           25,       'number of sampled frame per video')
   cmd:option('-numSegment',          3,        'number of segments per video')
   cmd:option('-nStacking',           10,       'number of stacked images for temporal-stream ConvNet')
   cmd:option('-tempAugmentation',    'false',  'temporal augmentation: randomly sampled frames')
   ------------- Training options --------------------
   cmd:option('-shuffle',         'true',       'Shuffle training data')
   cmd:option('-nEpochs',         0,            'Number of total epochs to run')
   cmd:option('-epochNumber',     1,            'Manual epoch number (useful on restarts)')
   cmd:option('-batchSize',       128,           'mini-batch size (1 = pure stochastic)')
   cmd:option('-testOnly',        'false',      'Run on validation set only')
   cmd:option('-tenCrop',         'false',      'Ten-crop testing')
   ------------- Checkpointing options ---------------
   cmd:option('-logName', 		  'model_RNN', 	'the name of your experiment')
   cmd:option('-save',            'checkpoints', 'Directory in which to save checkpoints')
   cmd:option('-resume',          'none',        'Resume from the latest checkpoint in this directory')
   ---------- Optimization options ----------------------
   cmd:option('-optimizer', 	  'adam', 	   'optimizer: sgd | adam | adamax | rmsprop')
   cmd:option('-LR',              5e-5,         'initial learning rate')
   cmd:option('-epochUpdateLR',   60, 			'learning rate decay per epochs')
   cmd:option('-momentum',        0,            'momentum')
   cmd:option('-weightDecay',     0,            'weight decay')
   ---------- Model options ----------------------------------
   cmd:option('-netType',         'rnn',        'Options: rnn')
   cmd:option('-lstm',            'true',       'Is LSTM being used')
   cmd:option('-inputSize',       4096,         'input dimensions for RNN')
   cmd:option('-hiddenSize',      '{512, 256}',      'Hidden dimensions for RNN: {1024,512}')
   cmd:option('-dropout', 		  0, 				'apply dropout after each recurrent layer')
   cmd:option('-bn', 			  'false', 		'Batch normalization')
   cmd:option('-uniform', 		  0.1, 			'initialize parameters using uniform distribution between -uniform and uniform. -1 means default initialization')
   cmd:option('-retrain',         'none',       'Path to model to retrain with')
   cmd:option('-optimState',      'none',       'Path to an optimState to reload from')
   ---------- Model options ----------------------------------
   cmd:option('-shareGradInput',  'false', 'Share gradInput tensors to reduce memory usage')
   cmd:option('-optnet',          'false', 'Not supported. Use optnet to reduce memory usage')
   cmd:option('-resetClassifier', 'false', 'Reset the fully connected layer for fine-tuning')
   cmd:option('-nClasses',         0,      'Number of classes in the dataset')
   cmd:text()

   local opt = cmd:parse(arg or {})

   opt.save = opt.save .. '/' .. opt.logName .. '-' .. 'split' .. opt.split .. '-' .. opt.tempAugmentation .. '-' .. opt.hiddenSize

   opt.shuffle = opt.shuffle ~= 'false'
   opt.tempAugmentation = opt.tempAugmentation ~= 'false'
   opt.bn = opt.bn ~= 'false'
   opt.hiddenSize = loadstring(" return "..opt.hiddenSize)()

   opt.lstm = opt.lstm ~= 'false'
   opt.testOnly = opt.testOnly ~= 'false'
   opt.tenCrop = opt.tenCrop ~= 'false'
   opt.shareGradInput = opt.shareGradInput ~= 'false'
   opt.optnet = opt.optnet ~= 'false'
   opt.resetClassifier = opt.resetClassifier ~= 'false'

   if not paths.dirp(opt.save) and not paths.mkdir(opt.save) then
      cmd:error('error: unable to create checkpoint directory: ' .. opt.save .. '\n')
   end

   opt.rgbData = opt.rgbData .. 'split' .. opt.split
   opt.flowData = opt.flowData .. 'split' .. opt.split

   if opt.dataset == 'ucf101' then
      opt.nClasses = 101
   elseif opt.dataset == 'hmdb51' then
      opt.nClasses = 51
   end

   if opt.tempAugmentation then
      print('-------- Temporal Augmentation --------')
   end

   if opt.dataset == 'ucf101' then
      -- Handle the most common case of missing -data flag
      local trainDir = paths.concat(opt.rgbData, 'train')
      if not paths.dirp(opt.rgbData) then
         cmd:error('error: missing ' .. opt.dataset .. ' RGB data directory')
      elseif not paths.dirp(trainDir) then
         cmd:error('error: ' .. opt.dataset .. ' missing `train` directory: ' .. trainDir)
      end
      trainDir = paths.concat(opt.flowData, 'train')
      if not paths.dirp(opt.flowData) then
         cmd:error('error: missing ' .. opt.dataset .. ' flow data directory')
      elseif not paths.dirp(trainDir) then
         cmd:error('error: ' .. opt.dataset .. ' missing `train` directory: ' .. trainDir)
      end
      opt.nEpochs = opt.nEpochs == 0 and 90 or opt.nEpochs
   else
      cmd:error('unknown dataset: ' .. opt.dataset)
   end

   if opt.precision == nil or opt.precision == 'single' then
      opt.tensorType = 'torch.CudaTensor'
   elseif opt.precision == 'double' then
      opt.tensorType = 'torch.CudaDoubleTensor'
   elseif opt.precision == 'half' then
      opt.tensorType = 'torch.CudaHalfTensor'
   else
      cmd:error('unknown precision: ' .. opt.precision)
   end

   if opt.resetClassifier then
      if opt.nClasses == 0 then
         cmd:error('-nClasses required when resetClassifier is set')
      end
   end
   if opt.shareGradInput and opt.optnet then
      cmd:error('error: cannot use both -shareGradInput and -optnet')
   end

   return opt
end

return M
