Association Modules
====================
These modules contain functions for the two 
main association methods in astro_ghost: Directional
Light Radius (DLR) and Gradient Ascent (GA). A brief summary
of each technique is given below. 

DLR
-----------------------
The module to associate transients using the 
Directional Light Radius (DLR) method outlined
in Gupta et al., 2013. The DLR method estimates 
the radius of each candidate host in the direction 
of the transient, normalizes each galaxy-transient
distance by this value, and takes the source with 
the lowest normalized distance as the true host.

.. automodule:: astro_ghost.DLR
   :members:
   :undoc-members:
   :show-inheritance:

gradientAscent
----------------------------------
The module to associate transients using the 
Gradient Ascent (GA) method outlined in Gagliano 
et al., 2021. The GA method downloads PS1 postage
stamps of the field, preprocesses the images to 
remove local structure such as HII regions, and 
then follows the 2D image gradients from the 
position of the transient to a local brightness
maximum. It then queries PS1 for sources near the 
final position. This is slower than DLR (because of 
the time needed to download large postage stamps), but 
can be more accurate for well-resolved low-redshift
hosts.

.. automodule:: astro_ghost.gradientAscent
   :members:
   :undoc-members:
   :show-inheritance:
