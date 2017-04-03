--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  UCF101 dataset loader
--

local image = require 'image'
local paths = require 'paths'
local t = require 'datasets/transforms'
local ffi = require 'ffi'

local M = {}
local UCF101Dataset = torch.class('rnn.UCF101Dataset', M)

function UCF101Dataset:__init(imageInfo, flowInfo, opt, split)
   self.imageInfo = imageInfo[split]
   self.opt = opt
   self.split = split
   self.dir = paths.concat(opt.rgbData, split)

   self.flowInfo = flowInfo[split]
   self.flowDir = paths.concat(opt.flowData, split)

   assert(paths.dirp(self.dir), 'directory does not exist: ' .. self.dir)
   assert(paths.dirp(self.flowDir), 'Optical flow directory does not exist: ' .. self.flowDir)

end

function UCF101Dataset:get(i)
   local rgbPath = ffi.string(self.imageInfo.featurePath[i]:data())
   local flowPath = ffi.string(self.flowInfo.featurePath[i]:data())

   rgbPath = paths.concat(self.dir, rgbPath)
   flowPath = paths.concat(self.flowDir, flowPath)

   local feature = self:_loadFeature(rgbPath, flowPath)
   local class = self.imageInfo.featureClass[i]

   return {
      input = torch.cat(feature.rgb, feature.flow, 2),
      target = class,
   }
end

function UCF101Dataset:_loadFeature(rgbPath, flowPath)
   local function getRandIndex(x)
      -- get random index from equally spaced snippests
      local randIndex = torch.LongTensor(x:size(1) - 1)
         for i = 1, x:size(1)-1 do
            randIndex[i] = torch.random(x[i], x[i+1]) 
         end

      return randIndex
   end

   local function videoSnippests(length)
      -- given the length of the video, get the index of feature vectors we want to sample from

      -- divide the video into several segments
      local segLength = (length - self.opt.nStacking + 1) / self.opt.nVideoSeg
      local snippetsIndex = torch.LongTensor(self.opt.nVideoSeg + 1):zero()

      -- get the start index of each snippets
      for j = 1, self.opt.nVideoSeg do
         snippetsIndex[j] = self.opt.nStacking/2 + torch.floor(segLength*(j-1))
      end

      -- BUG: unsolved
      -- because of the frame difference between RGB and optical flow
      -- snippetsIndex[-1] = length - self.opt.nStacking/2 - 1
      snippetsIndex[-1] = length - self.opt.nStacking/2

      -- get random index from equally spaced snippests
      local featureIndex
      if self.opt.tempAugmentation then
         featureIndex = getRandIndex(snippetsIndex)
      else
         featureIndex = snippetsIndex[{{1, self.opt.nVideoSeg}}] -- no random for testing
      end

      return featureIndex
   end

   local function recursiveIsNaN(tensor)
      local isNaN = false
      if torch.type(tensor) == 'table' then
         for k,v in pairs(tensor) do
            isNaN = self:recursiveIsNaN(v)
            if isNaN then break end
         end
      else
         local _ = require 'moses'
         isNaN = _.isNaN(tensor:sum())
      end
      return isNaN
   end

   local function grabFeatures(feature, indexList)
      -- given a list of index, grab the feature vectors according to the list
      local length = indexList:size(1)

      -- grab the first feature vector
      local sampledFeature = feature[{{indexList[1]}}]

      -- stack the following feature vectors
      for i = 2, indexList:size(1) do
         sampledFeature = torch.cat(sampledFeature, feature[{{indexList[i]}}], 1)
      end

      -- print('checking NaN...')
    --   if recursiveIsNaN(sampledFeature) then
    --      local nan_mask = sampledFeature:ne(sampledFeature)
    --      print(sampledFeature)
    --      print(nan_mask)
    --      error('found NaN')
    --   end

      return sampledFeature
   end

--    print(collectgarbage("count")*1024)


   local obj_rgb = torch.load(rgbPath)
   local frameLevelFeatures_rgb = obj_rgb.mat

   local obj_flow = torch.load(flowPath)
   local frameLevelFeatures_flow = obj_flow.mat
  
   local featureIndex = videoSnippests(frameLevelFeatures_rgb:size(1))

   local rgbFeature = grabFeatures(frameLevelFeatures_rgb, featureIndex)
   local flowFeature = grabFeatures(frameLevelFeatures_flow, featureIndex)

   return {
      rgb = rgbFeature, 
      flow = flowFeature,
   }
end

function UCF101Dataset:size()
   return self.imageInfo.featureClass:size(1)
end

return M.UCF101Dataset
