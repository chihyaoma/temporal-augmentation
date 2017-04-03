--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Script to compute list of filenames and classes
--
--  This generates a file gen/opt.dataset.t7 which contains the list of all
--  training and validation features and their classes. This script also
--  works for other datasets arragned with the same layout.
--

local sys = require 'sys'
local ffi = require 'ffi'

local M = {}

local function findClasses(dir)
   local dirs = paths.dir(dir)
   table.sort(dirs)

   local classList = {}
   local classToIdx = {}
   for _ ,class in ipairs(dirs) do
      if not classToIdx[class] and class ~= '.' and class ~= '..' and class ~= '.DS_Store' then
         table.insert(classList, class)
         classToIdx[class] = #classList
      end
   end

   return classList, classToIdx
end

local function findFeatures(dir, classToIdx)
   local featurePath = torch.CharTensor()
   local featureClass = torch.LongTensor()

   ----------------------------------------------------------------------
   -- Options for the GNU and BSD find command
   local extensionList = {'t7'}
   local findOptions = ' -iname "*.' .. extensionList[1] .. '"'
   for i=2,#extensionList do
      findOptions = findOptions .. ' -o -iname "*.' .. extensionList[i] .. '"'
   end

   -- Find all the images using the find command
--    local f = io.popen('find -L ' .. dir .. findOptions)
   local f = io.popen('find -L ' .. dir .. findOptions .. ' | sort')

   local maxLength = -1
   local featurePaths = {}
   local featureClasses = {}

   -- Generate a list of all the images and their class
   while true do
      local line = f:read('*line')
      if not line then break end

      local className = paths.basename(paths.dirname(line))
      local filename = paths.basename(line)
      local path = className .. '/' .. filename

      local classId = classToIdx[className]
      assert(classId, 'class not found: ' .. className)

      table.insert(featurePaths, path)
      table.insert(featureClasses, classId)

      maxLength = math.max(maxLength, #path + 1)
   end

   f:close()

   -- Convert the generated list to a tensor for faster loading
   local nFeatures = #featurePaths
   local featurePath = torch.CharTensor(nFeatures, maxLength):zero()
   for i, path in ipairs(featurePaths) do
      ffi.copy(featurePath[i]:data(), path)
   end

   local featureClass = torch.LongTensor(featureClasses)
   return featurePath, featureClass
end

function M.exec(opt, cacheFile)
   -- find the image path names
   local featurePath = torch.CharTensor()  -- path to each image in dataset
   local featureClass = torch.LongTensor() -- class index of each image (class index in self.classes)

   local trainDir = paths.concat(opt.flowData, 'train')
   local valDir = paths.concat(opt.flowData, 'val')
   assert(paths.dirp(trainDir), 'train directory not found: ' .. trainDir)
   assert(paths.dirp(valDir), 'val directory not found: ' .. valDir)

   print("=> Generating list of features")
   local classList, classToIdx = findClasses(trainDir)

   print(" | finding all validation features")
   local valFeaturePath, valFeatureClass = findFeatures(valDir, classToIdx)

   print(" | finding all training features")
   local trainFeaturePath, trainFeatureClass = findFeatures(trainDir, classToIdx)

   local info = {
      basedir = opt.flowData,
      classList = classList,
      train = {
         featurePath = trainFeaturePath,
         featureClass = trainFeatureClass,
      },
      val = {
         featurePath = valFeaturePath,
         featureClass = valFeatureClass,
      },
   }

   print(" | saving list of features to " .. cacheFile)
   torch.save(cacheFile, info)
   return info
end

return M
