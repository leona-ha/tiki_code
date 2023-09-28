import warnings
import geopandas as gpd

# We used functions from
# Zhou, Shuai, Yanling Li, Guangqing Chi, Junjun Yin, Zita Oravecz, Yosef Bodovski, Naomi P. Friedman, 
# Scott I. Vrieze, and Sy-Miin Chow. 2021. "GPS2space: An Open-source Python Library for Spatial 
# Measure Extraction from GPS Data." Journal of Behavioral Data Science. 1(2): 127–155.

import warnings
import geopandas as gpd
import pandas as pd

# %%
def df_to_gdf(df, x='long', y='lat'):
	"""
	Transform raw Lat/Long data to GeoDataFrame
	
	Parameters
	==========
	df: DataFrame
	x: Latitude
	y: Longitude
	
	Returns
	=======
	gdf: Point GeoDataFrame (unprojected)
	"""
	gdf = gpd.GeoDataFrame(df,
			       geometry=gpd.points_from_xy(df[x], df[y]),
			       crs=("epsg:4326"))
	return gdf



def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r

# activity space based on buffer
def buffer_space(gdf, dist=0, dissolve='week', proj=2163):
	"""
	Perform activity space based on buffer method

	Parameters
	==========
	gdf: GeoDataFrame
	dist: buffer distance in meters
	dissolve: level of aggregating points to form polygon

	Returns
	=======
	gdf: Polygon GeoDataFrame (user-defined projection)
	"""
	# gdf.crs = ("epsg:4326")
	gdf = gdf.to_crs(epsg=proj)
	gdf['geometry'] = gdf.geometry.buffer(dist)
	polys = gdf.dissolve(by=[dissolve]).reset_index()
	polys['buff_area'] = polys['geometry'].area
	return polys

# activity space based on convex
def convex_space(gdf, group='week', proj=2163):
	"""
	Perform activity space based on convex hull method.

	Parameters
	==========
	gdf: GeoDataFrame
	group: level of aggregating points to form polygon

	Returns
	=======
	gdf: Polygon GeoDataFrame (user-defined projection)
	"""
	# Make a separate DataFrame from gdf, but remove geometry column
	# And drop duplicates in terms of the "group" parameter
	df_temp = gdf.drop('geometry', 1)
	# print(df_temp.head())
	df = df_temp.drop_duplicates(group)

	# Obtain the convex hull activity space
	gdf = gdf.to_crs(epsg=proj)
	groups = gdf.groupby(group)
	convex = groups.geometry.apply(lambda x: x.unary_union.convex_hull)
	convex = gpd.GeoDataFrame(convex.reset_index())
	convex['convex_area'] = convex['geometry'].area

	# Merge convex with gdf_temp on group to get the columns from original gdf
	convex = convex.merge(df, on=group)

	return convex

# activity space based on concave