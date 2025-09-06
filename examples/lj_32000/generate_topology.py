num_atoms = 32000
with open("topology.top", "w") as f:
    f.write("; Simple Lennard-Jones topology file\n")
    f.write("; Generated for 32000 particles\n\n")
    f.write("[ atomtypes ]\n")
    f.write("; name mass charge sigma epsilon\n")
    f.write("   Ar  39.948  0.000  3.405  0.997\n\n")
    f.write("[ atoms ]\n")
    f.write("; id type mass charge\n")
    for i in range(1, num_atoms + 1):
        f.write(f"{i:5d}   Ar   39.948    0.000\n")
print("topology.top 생성 완료 (LJ 32000, Ar 형식)")
