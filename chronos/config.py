from pkg_resources import resource_filename
from os.path import join
#import getpass
#user = getpass.getuser()

DATA_PATH = resource_filename(__name__, "data")
#FITSOUTDIR = join("/home", user, "data/transit")
FITS_OUTDIR = join(DATA_PATH, "transit")
