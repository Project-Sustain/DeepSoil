import socket
import geopandas as gpd
import mercantile,fiona
import os, osr

# GET QUADHASH TILE OF A GIVEN COORDINATE
def get_quad_tile(lat, lon, precision):
    ret = mercantile.tile(lon,lat,precision)
    return ret

def get_quad_key_from_tile(x, y, zoom):
    return mercantile.quadkey(x, y, zoom)

# GIVEN A QUAD_KEY, GET THE CORRESPONDING QUAD TILE
def get_tile_from_key(key):
    return mercantile.quadkey_to_tile(key)

# GET QUADHASH STRING OF A GIVEN COORDINATE
def get_quad_key(lat, lon, zoom):
    tile = get_quad_tile(lat, lon, precision=zoom)
    return get_quad_key_from_tile(tile.x, tile.y, tile.z)

#GIVEN A ZOOM LEVEL, WHAT IS THE MAX POSSIBLE TILE NUMBER HERE?
def get_max_possible_xy(zoom):
    if zoom == 0:
        return 0
    return 2**zoom-1

# GIVEN A TILE, VERIFY IT IS VALID
def validate_tile(tile):
    max_xy = get_max_possible_xy(tile.z)
    if tile.x > max_xy or tile.x < 0 or tile.y > max_xy or tile.y < 0:
        return False
    return True

# GIVEN A BOX, FIND ALL TILES THAT LIE INSIDE THAT COORDINATE BOX
def find_all_inside_box(lat1, lat2, lon1, lon2, zoom):
    all_tiles,all_tiles_quadhash,bounding_boxes = [], [],[]
    top_left_quad_tile = get_quad_tile(lat1, lon1, zoom)
    bottom_right_quad_tile = get_quad_tile(lat2, lon2, zoom)

    x1 = top_left_quad_tile.x
    x2 = bottom_right_quad_tile.x
    y1 = top_left_quad_tile.y
    y2 = bottom_right_quad_tile.y

    for i in range(x1, x2+1):
        for j in range(y1,y2+1):
            all_tiles.append(mercantile.Tile(x=i,y=j,z=zoom))
            info = mercantile.Tile(x=i,y=j,z=zoom)
            qk = get_quad_key_from_tile(info.x, info.y, info.z)
            all_tiles_quadhash.append(qk)
            bounding_boxes.append(get_bounding_lng_lat(qk))
    return all_tiles_quadhash,bounding_boxes

#GIVEN A TILE, FIND THE SMALLER TILES THAT LIE INSIDE
def get_inner_tiles(tile_string):
    combos = range(4)
    children = []
    for i in combos:
        t_s = tile_string+str(i)
        children.append(get_tile_from_key(t_s))
    return children

#GIVEN A QUAD_TILE, GET ITS LAT-LNG BOUNDS
def get_bounding_lng_lat(tile_key):
    tile = get_tile_from_key(tile_key)
    bounds = mercantile.bounds(tile)
    return [bounds.west, bounds.east, bounds.north, bounds.south]

def create_shp_file():
    states = gpd.read_file('./cb_2018_us_state_500k/cb_2018_us_state_500k.shp')
    states = states.to_crs("EPSG:4326")
    ignore_states = [13, 27, 36, 37, 38, 44, 42, 45]
    count = -1
    for ind, row in states.iterrows():
        if ind in ignore_states:
            continue
        count += 1
        states_on_machine = 176 + (count % 48)
        if states_on_machine == 177:
            states_on_machine = 224
        if states_on_machine == int(socket.gethostname().split("-")[1]):
            download_for_state = row['NAME']
            break

    print("Processing", download_for_state, "on", socket.gethostname())
    state_info = states[states['NAME'] == download_for_state]

    schema = {
        'geometry': 'Polygon',
        'properties': [('Quadkey', 'str')]
    }

    area_of_interest_lat1, area_of_interest_lat2 = state_info["geometry"].bounds['maxy'].iloc[0],\
        state_info["geometry"].bounds['miny'].iloc[0]
    area_of_interest_lon1, area_of_interest_lon2 = state_info["geometry"].bounds['minx'].iloc[0],\
        state_info["geometry"].bounds['maxx'].iloc[0]

    zoom = [9, 12, 14]
    for z in zoom:
        os.makedirs('./quadshape_' + str(z) + '_' + download_for_state.replace(" ", "_"), exist_ok=True)
        polyShp = fiona.open('./quadshape_' + str(z) + '_' + download_for_state.replace(" ", "_") + '/quadhash.shp', mode='w', driver='ESRI Shapefile',
                             schema=schema, crs="EPSG:4326")

        quads, bounds = find_all_inside_box(area_of_interest_lat1, area_of_interest_lat2, area_of_interest_lon1,
                                            area_of_interest_lon2, zoom=z)

        for i in range(len(quads)):
            xyList = []
            xyList.append((bounds[i][0],bounds[i][2]))
            xyList.append((bounds[i][1],bounds[i][2]))
            xyList.append((bounds[i][1],bounds[i][3]))
            xyList.append((bounds[i][0],bounds[i][3]))
            xyList.append((bounds[i][0],bounds[i][2]))

            rowDict = {
                'geometry': {'type': 'Polygon',
                             'coordinates': [xyList]},  # Here the xyList is in brackets
                'properties': {'Quadkey': quads[i]},
            }
            polyShp.write(rowDict)
        print("Shape file generated succesfully with no. of rows: ", i+1, " at zoom level: ", z)
        polyShp.close()

if __name__ == '__main__':
    create_shp_file()