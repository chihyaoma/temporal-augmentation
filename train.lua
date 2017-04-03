--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  The training loop and learning rate schedule
--

local optim = require 'optim'
local sys = require 'sys'
local optim = require 'optim'

local M = {}
local Trainer = torch.class('rnn.Trainer', M)

function Trainer:__init(model, criterion, opt, optimState)
   self.model = model
   self.criterion = criterion
   self.optimState = optimState or {
      learningRate = opt.LR,
      learningRateDecay = 0.0,
      momentum = opt.momentum,
      nesterov = true,
      dampening = 0.0,
      weightDecay = opt.weightDecay,
   }
   self.opt = opt
   self.params, self.gradParams = model:getParameters()

   self.testLogger = optim.Logger(paths.concat(opt.save,'test.log'))
end

function Trainer:train(epoch, dataloader)
   -- Trains the model for a single epoch
   self.optimState.learningRate = self:learningRate(epoch)

   local timer = torch.Timer()
   local dataTimer = torch.Timer()

   local function feval()
      return self.criterion.output, self.gradParams
   end

   local trainSize = dataloader:size()
   local top1Sum, top5Sum, lossSum = 0.0, 0.0, 0.0
   local N = 0

   print('=> Training epoch # ' .. epoch)
   -- set the batch norm to training mode
   self.model:training()
   for n, sample in dataloader:run(self.opt) do
      local dataTime = dataTimer:time().real

      -- Copy input and target to the GPU
      self:copyInputs(sample)

      if sample.input:size(1) == self.opt.batchSize then
         -- to avoid possible bug when batchsize is inconsistent 
         -- the rnn model:remember('both') will break
         -- skip the last batch
    

         local inputsSegments = {}
         local segmentBasis = math.floor(self.input:size(2)/self.opt.numSegment)
         for s = 1, self.opt.numSegment do
            table.insert(inputsSegments, self.input[{{},{segmentBasis*(s-1) + 1,segmentBasis*s}, {}}])
         end

         local output = self.model:forward(inputsSegments):float()
         local batchSize = output:size(1)
         local loss = self.criterion:forward(self.model.output, self.target)

         self.model:zeroGradParameters()
         self.criterion:backward(self.model.output, self.target)
         self.model:backward(inputsSegments, self.criterion.gradInput)

         if self.opt.optimizer == 'sgd' then
            optim.sgd(feval, self.params, self.optimState)
         elseif self.opt.optimizer == 'adam' then
            optim.adam(feval, self.params, self.optimState)
         elseif self.opt.optimizer == 'adamax' then
            optim.adamax(feval, self.params, self.optimState)
         elseif self.opt.optimizer == 'rmsprop' then
            optim.rmsprop(feval, self.params, self.optimState)
         end 

         local top1, top5 = self:computeScore(output, sample.target, 1)
         top1Sum = top1Sum + top1*batchSize
         top5Sum = top5Sum + top5*batchSize
         lossSum = lossSum + loss*batchSize
         N = N + batchSize

         print(('Best: %.3f | Epoch: [%d][%d/%d]    Time %.3f  Data %.3f  Err %1.4f  top1 %7.3f  top5 %7.3f'):format(
            100-bestTop1, epoch, n, trainSize, timer:time().real, dataTime, loss, top1, top5))

         -- check that the storage didn't get changed due to an unfortunate getParameters call
         assert(self.params:storage() == self.model:parameters()[1]:storage())

         timer:reset()
         dataTimer:reset()
      end
   end

   return top1Sum / N, top5Sum / N, lossSum / N
end

function Trainer:test(epoch, dataloader)
   -- Computes the top-1 and top-5 err on the validation set

   local timer = torch.Timer()
   local dataTimer = torch.Timer()
   local size = dataloader:size()

   local nCrops = self.opt.tenCrop and 10 or 1
   local top1Sum, top5Sum = 0.0, 0.0
   local N = 0

   self.model:evaluate()
   for n, sample in dataloader:run(self.opt) do
      local dataTime = dataTimer:time().real

      if sample.input:size(1) ~= self.opt.batchSize then
         -- to avoid problem when batchsize is inconsistent 
         -- the rnn model:remember('both') will break
         -- Padding the input and target to batchsize
         self.input:fill(0)
         self.input[{{1,sample.input:size(1)}}]:copy(sample.input)
         self.target[{{1,sample.target:size(1)}}]:copy(sample.target)
      else
         -- Copy input and target to the GPU
         self:copyInputs(sample)
      end  

      local inputsSegments = {}
      local segmentBasis = math.floor(self.input:size(2)/self.opt.numSegment)
      for s = 1, self.opt.numSegment do
         table.insert(inputsSegments, self.input[{{},{segmentBasis*(s-1) + 1,segmentBasis*s}, {}}])
      end

      local output = self.model:forward(inputsSegments):float()

      if sample.input:size(1) ~= self.opt.batchSize then
         -- to avoid possible bug when batchsize is inconsistent 
         -- the rnn model:remember('both') will break
         output = output[{{1, sample.input:size(1)}}]
      end

      local batchSize = output:size(1) / nCrops
      local loss = self.criterion:forward(self.model.output, self.target)

      local top1, top5 = self:computeScore(output, sample.target, nCrops)
      top1Sum = top1Sum + top1*batchSize
      top5Sum = top5Sum + top5*batchSize
      N = N + batchSize

      print(('Best: %.3f | Test: [%d][%d/%d]    Time %.3f  Data %.3f  top1 %7.3f (%7.3f)  top5 %7.3f (%7.3f)'):format(
         100-bestTop1, epoch, n, size, timer:time().real, dataTime, top1, top1Sum / N, top5, top5Sum / N))

      timer:reset()
      dataTimer:reset()
   end
   self.model:training()

   print((' * Finished epoch # %d     top1: %7.3f  top5: %7.3f\n'):format(
      epoch, top1Sum / N, top5Sum / N))
   print(sys.COLORS.red .. '==> Best testing accuracy = ' .. 100-bestTop1 .. '%')
	self.testLogger:add{['epoch'] = epoch-1, ['top-1 accuracy'] = 100 - top1Sum / N}

   return top1Sum / N, top5Sum / N
end

function Trainer:computeScore(output, target, nCrops)
   if nCrops > 1 then
      -- Sum over crops
      output = output:view(output:size(1) / nCrops, nCrops, output:size(2))
         --:exp()
         :sum(2):squeeze(2)
   end

   -- Coputes the top1 and top5 error rate
   local batchSize = output:size(1)

   local _ , predictions = output:float():topk(5, 2, true, true) -- descending

   -- Find which predictions match the target
   local correct = predictions:eq(
      target:long():view(batchSize, 1):expandAs(predictions))

   -- Top-1 score
   local top1 = 1.0 - (correct:narrow(2, 1, 1):sum() / batchSize)

   -- Top-5 score, if there are at least 5 classes
   local len = math.min(5, correct:size(2))
   local top5 = 1.0 - (correct:narrow(2, 1, len):sum() / batchSize)

   return top1 * 100, top5 * 100
end

local function getCudaTensorType(tensorType)
  if tensorType == 'torch.CudaHalfTensor' then
     return cutorch.createCudaHostHalfTensor()
  elseif tensorType == 'torch.CudaDoubleTensor' then
    return cutorch.createCudaHostDoubleTensor()
  else
     return cutorch.createCudaHostTensor()
  end
end

function Trainer:copyInputs(sample)
   -- Copies the input to a CUDA tensor, if using 1 GPU, or to pinned memory,
   -- if using DataParallelTable. The target is always copied to a CUDA tensor
   self.input = self.input or (self.opt.nGPU == 1
      and torch[self.opt.tensorType:match('torch.(%a+)')]()
      or getCudaTensorType(self.opt.tensorType))
   self.target = self.target or (torch.CudaLongTensor and torch.CudaLongTensor())
   self.input:resize(sample.input:size()):copy(sample.input)
   self.target:resize(sample.target:size()):copy(sample.target)
end

function Trainer:learningRate(epoch)
   -- Training schedule
   local decay = 0
   if self.opt.dataset == 'ucf101' then
      decay = math.floor((epoch - 1) / 100)
   elseif self.opt.dataset == 'hmdb51' then
      decay = math.floor((epoch - 1) / 100)
   else
      error('unknown dataset: ' .. self.opt.dataset)
   end
   return self.opt.LR * math.pow(0.1, decay)
end

return M.Trainer
