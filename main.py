from sklearn import tree
from collections import OrderedDict
from json import loads

rawDataResults = loads(open('./results.json', 'r').read())
rawDataDistros = loads(open('./distro.json', 'r').read())

groupedResults = {}
seenTags = []

# map the tags per result
for result in rawDataResults:
  if result["tag"] not in seenTags:
    seenTags.append(result["tag"])
  if result["resultid"] not in groupedResults:
    groupedResults[result["resultid"]] = {}
  if result["tag"] not in groupedResults[result["resultid"]]:
    groupedResults[result["resultid"]][result["tag"]] = 1*int(result["weight"])
    if int(result["isNegative"]) == 1:
      groupedResults[result["resultid"]][result["tag"]] = groupedResults[result["resultid"]][result["tag"]]*-1
# search for missing tags, append them if needed
for resultId, tags in groupedResults.items():
  missingTags = []
  for tag in seenTags:
    if tag not in tags:
      missingTags.append(tag)
  for tag in missingTags:
    tags[tag] = 0

# arrange all tags in correct (alphabetical) order
for resultId, _ in groupedResults.items():  
  sortedKeys = sorted(groupedResults[resultId])
  orderedTags = OrderedDict()
  for key in sortedKeys:
    orderedTags[key] = groupedResults[resultId][key]
  groupedResults[resultId] = orderedTags

# arrange distros according their result to the tags
distroTagsResults = {}
for resultId, tags in groupedResults.items():
  distroRatings = {}
  for distro in rawDataDistros:
    distroTags = loads(distro["tags"])
    tagScores = {}
    distroScore = 0
    for tag, score in tags.items():
      if tag in distroTags:
        distroScore = distroScore + score
      if "!" + tag in distroTags:
        distroScore = 0 
        break
    if distroScore > 0:
      distroRatings[distro["name"]] = distroScore
  rankedDistros = sorted(distroRatings.items(), key=lambda kv: kv[1],reverse=True)
  for distroName, distroScore in rankedDistros:
    distroTagsResults[resultId+"#"+distroName] = []
    for _, value in tags.items():
      distroTagsResults[resultId+"#"+distroName].append(value)

# separate distroTagsResults in X and y

X = []
Y = []
for key, distroTagsResult in distroTagsResults.items():
  distroName = key.split("#")[1]
  Y.append(distroName)
  X.append(distroTagsResult)

clf = tree.DecisionTreeClassifier(max_depth=3)
clf = clf.fit(X, Y)


userInputHumanReadable = OrderedDict([('all-like', 0), ('container', 0), ('fromsource', 0), ('help-community', 1), ('help-wiki', 1), ('installation-base', 0), ('installation-full', 2), ('installation-hdd', 1), ('installation-live', 0), ('installation-usb', -1), ('installer-defaults-wanted', 2), ('installer-no-defaults-wanted', 0), ('license-free', 0), ('license-unfree-if-needed', 1), ('linux-advanced', 0), ('linux-beginner', 2), ('linux-expert', 0), ('mac-like', 0), ('multipackage', 0), ('no-systemd', 0), ('pay-nothing', 2), ('pay-price', 0), ('pc-advanced', 0), ('pc-beginner', 0), ('pc-expert', 3), ('pc-old', 0), ('pc-up-to-date', 2), ('privacy-online-not-okay', 0), ('privacy-online-okay', 1), ('programs-graphical', 4), ('programs-shell', 0), ('systemd', 2), ('updates-stable', 2), ('updates-unstable', 0), ('usage-anon', 0), ('usage-daily', 1), ('usage-gaming', 1), ('usage-rescue', -1), ('usage-science', 0), ('usage-usb', -1), ('ux-closed', 2), ('ux-undecided', 0), ('windows-like', 0)])

userInput = [[]]
for tag, score in userInputHumanReadable.items():
  userInput[0].append(score)

print(userInput)

print(clf.predict(userInput), clf.predict_proba(userInput))
for target in set(Y):
  print(target, "->" , clf.score(userInput, [target]))