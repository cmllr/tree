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
for key, distroTagsResults in distroTagsResults.items():
  distroName = key.split("#")[1]
  Y.append(distroName)
  X.append(distroTagsResults)


clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)

userInput = [[1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 2, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0]]

print(userInput, clf.predict(userInput), clf.predict_proba(userInput))
for target in set(Y):
  print(target, "->" , clf.score(userInput, [target]))