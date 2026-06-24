import uproot
import matplotlib.pyplot as plt

f = uproot.open("background_toy.root")
tree = f["background"]
data = tree.arrays(library="pd")

plt.figure()
plt.hist(data["cosThetaK"], bins=50)
plt.xlabel("cosThetaK")
plt.ylabel("Events")
plt.title("Background toy: cosThetaK")
plt.show()

plt.figure()
plt.hist(data["cosThetaL"], bins=50)
plt.xlabel("cosThetaL")
plt.ylabel("Events")
plt.title("Background toy: cosThetaL")
plt.show()

print(tree.keys())