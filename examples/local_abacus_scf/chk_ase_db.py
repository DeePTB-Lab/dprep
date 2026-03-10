from ase.visualize import view
from ase.db import connect
from ase.io import read, write
import os


with connect('test_4_structures.db') as tgt_db:
    print(tgt_db.count())
    for a_row in tgt_db.select():
        an_atoms = a_row.toatoms()
        view(an_atoms)
        a=1

