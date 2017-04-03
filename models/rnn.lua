--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  The ResNet model definition
--

local nn = require 'nn'
require 'cunn'
require 'rnn'
require 'dpnn'

local function createModel(opt)

   local function tempSeg_BN_max_LSTM_BN_FC()
      -- model: BN + Max + LSTM + BN + FC
      local model = nn.Sequential()
      local inputSize = opt.inputSize

      local split = nn.ParallelTable()
      for i = 1, opt.numSegment do 
         split:add(nn.SplitTable(2,1))
      end
      model:add(split)

      local pBN = nn.ParallelTable()
      for i = 1, opt.numSegment do 
         pBN:add(nn.Sequencer(nn.BatchNormalization(inputSize)))
      end
      model:add(pBN)

      local mergeTable1 = nn.ParallelTable()
      for i = 1, opt.numSegment do 
         mergeTable1:add(nn.MapTable(nn.Unsqueeze(1)))
      end
      model:add(mergeTable1)

      local mergeTable2 = nn.ParallelTable()
      for i = 1, opt.numSegment do 
         mergeTable2:add(nn.JoinTable(1))
      end
      model:add(mergeTable2)

      local poolingTable = nn.ParallelTable()

      for i = 1, opt.numSegment do 
         poolingTable:add(nn.Max(1, -1))
      --   poolingTable:add(nn.Mean(1, -1))
      end
      model:add(poolingTable)

      for i,hiddenSize in ipairs(opt.hiddenSize) do 
         model:add(nn.Sequencer(nn.LSTM(inputSize, hiddenSize)))
         inputSize = hiddenSize
      end

      model:add(nn.SelectTable(-1))
    

      model:add(nn.BatchNormalization(inputSize))
         
      -- Dropout layer
      if opt.dropout > 0 then 
         model:add(nn.Dropout(opt.dropout))
      end
      model:add(nn.Linear(inputSize, opt.nClasses))

      return model
   end

   local function tempSeg_BN_LSTM_concat_FC()
      local inputSize = opt.inputSize
      local model = nn.Sequential()

      local split = nn.ParallelTable()
      for i = 1, opt.numSegment do 
         split:add(nn.SplitTable(2,1))
      end
      model:add(split)

      local pBN = nn.ParallelTable()
      for i = 1, opt.numSegment do 
         pBN:add(nn.Sequencer(nn.BatchNormalization(inputSize)))
      end
      model:add(pBN)

      for i,hiddenSize in ipairs(opt.hiddenSize) do 
         local lstmTable = nn.ParallelTable()
         for i = 1, opt.numSegment do 
            lstmTable:add(nn.Sequencer(nn.LSTM(inputSize, hiddenSize)))
         end
         model:add(lstmTable)
         inputSize = hiddenSize
      end

      local s1 = nn.ParallelTable()
      for i = 1, opt.numSegment do 
         s1:add(nn.SelectTable(-1))
      end
      model:add(s1)
      
      model:add(nn.JoinTable(2))      
      inputSize = inputSize * opt.numSegment

      model:add(nn.BatchNormalization(inputSize))

      -- Dropout layer
      if opt.dropout > 0 then 
         model:add(nn.Dropout(opt.dropout))
      end
      model:add(nn.Linear(inputSize, opt.nClasses))

      return model
   end

   local model = tempSeg_BN_LSTM_concat_FC()

   if opt.uniform > 0 then
      for k,param in ipairs(model:parameters()) do
         param:uniform(-opt.uniform, opt.uniform)
      end
   end

   -- will recurse a single continuous sequence
   model:remember(opt.lstm and 'both' or 'eval')

   model:type(opt.tensorType)

   print(model)

   return model
end

return createModel
