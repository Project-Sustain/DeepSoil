import socket
import os
from modis_tools.auth import ModisSession
from modis_tools.resources import CollectionApi, GranuleApi
from modis_tools.granule_handler import GranuleHandler
from datetime import datetime, timedelta
import gdal
import subprocess
import glob

username = "paahukh22"
password = "Paahuni@1234"

def download_lai_automatically():
    download_path = "/s/" + socket.gethostname() + "/b/nobackup/galileo/sm_predictions/input_datasets/MCD15A3H/raw/"
    session = ModisSession(username=username, password=password)
    collection_client = CollectionApi(session=session)
    collections = collection_client.query(short_name="MCD15A3H", version="061")

    granule_client = GranuleApi.from_collection(collections[0], session=session)

    # USA bounding box lat/lon
    usa_bbox = [-126.4745, 25.6415, -66.0062, 49.2678]

    current_date = datetime.now()
    start_date = (current_date - timedelta(days=4)).strftime("%Y-%m-%d")

    usa_granules = granule_client.query(start_date=start_date, bounding_box=usa_bbox)

    file_paths = GranuleHandler.download_from_granules(usa_granules, modis_session=session, path=download_path)
    print(file_paths)

def doy_yymmdd(year, day_of_year):
    target_date = datetime(year, 1, 1) + timedelta(day_of_year - 1)
    return target_date.strftime("%Y%m%d")

def convert_h5_to_tif():
    input_path = "/s/" + socket.gethostname() + "/b/nobackup/galileo/sm_predictions/input_datasets/MCD15A3H/raw/"
    output_path = "/s/" + socket.gethostname() + "/b/nobackup/galileo/sm_predictions/input_datasets/MCD15A3H/tifs/"

    total = len(os.listdir(input_path))
    count = 0
    for f in os.listdir(input_path):
        count += 1
        print("\nProcessing", count , "/", total)
        yymmdd = doy_yymmdd(int(f.split(".")[1][1:5]), int(f.split(".")[1][5:]))
        new_file_name = output_path + yymmdd + "_" + f.split(".")[2]+ ".tif"
        if os.path.exists(new_file_name):
            continue
        hdf_file = gdal.Open(input_path + f)
        subdatasets = hdf_file.GetSubDatasets()
        laifile = subdatasets[1][0]

        command = "gdalwarp -of GTIFF -s_srs '+proj=sinu +R=6371007.181 +nadgrids=@null +wktext' -r cubic -t_srs '+proj=longlat +datum=WGS84 +no_defs' " +\
                  laifile + " " + new_file_name

        subprocess.call(command, shell=True)

def transfer_tifs_to_all_machines():
    in_path = "/s/" + socket.gethostname() + "/b/nobackup/galileo/sm_predictions/input_datasets/MCD15A3H/tifs/"
    for send_to_lattice in range(176, 224):
        if send_to_lattice == int(socket.gethostname().split("-")[1]):
            continue
        out_path = "/s/lattice-" + str(send_to_lattice) + "/b/nobackup/galileo/sm_predictions/input_datasets/MCD15A3H/tifs/"

        files_to_transfer = os.listdir(in_path)

        for file_name in files_to_transfer:
            source_file = os.path.join(in_path, file_name)
            command = ['scp', source_file, 'paahuni@lattice-' + str(send_to_lattice) + ":" + out_path]
            subprocess.run(command)

        print("Sent to lattice-", send_to_lattice)

def chop_in_quadhash():
    quadhash_dir = next(d for d in os.listdir() if os.path.isdir(d) and d.startswith("quadshape_12_"))
    quadhashes = gpd.read_file(os.path.join(quadhash_dir, 'quadhash.shp'))

    in_path = "/s/" + socket.gethostname() + "/b/nobackup/galileo/sm_predictions/input_datasets/MCD15A3H/tifs/"
    out_path = "/s/" + socket.gethostname() + "/b/nobackup/galileo/sm_predictions/input_datasets/MCD15A3H/split_14/"

    count = 0
    total = len(quadhashes)
    for ind, row in quadhashes.iterrows():
        count+=1
        print("processing:  ", count, "/", total)
        poly,qua = row["geometry"], row["Quadkey"]
        os.makedirs(out_path + qua, exist_ok=True)
        bounds = list(poly.exterior.coords)
        window = (bounds[0][0], bounds[0][1], bounds[2][0], bounds[2][1])

        for f in os.listdir(in_path):
            gdal.Translate(out_path + qua + '/' + f,  in_path + f, projWin=window)

            x = gdal.Open(out_path + qua + '/' + f).ReadAsArray()
            if np.min(x) > 200 and np.max(x) > 200:
                os.remove(out_path + qua + '/' + f)
    remove_empty_folders()

def remove_empty_folders():
    in_path = "/s/" + socket.gethostname() + "/b/nobackup/galileo/sm_predictions/input_datasets/MCD15A3H/split_14/"
    tot = len(os.listdir(in_path))
    count = 0
    for q in os.listdir(in_path):
        if len(os.listdir(in_path + q)) == 0:
            print("No files in :", q)
            count += 1
            os.rmdir(in_path + q)
    print(count,"/",tot)

def merge_geotifs_for_quadhash():
    output_path_merged ="/s/" + socket.gethostname() + "/b/nobackup/galileo/sm_predictions/input_datasets/MCD15A3H/split_14/"
    no_data_value = 255

    for qua in os.listdir(output_path_merged):
        data_dic = {}
        for f in os.listdir(output_path_merged+qua):
            if "_" not in f:
                continue
            date_f = f.split("_")[0]
            if date_f in data_dic:
                files = data_dic.get(date_f)
                files.append(output_path_merged + qua + "/" + f)
                data_dic[date_f] = files
            else:
                data_dic[date_f] = [output_path_merged + qua + "/" + f]

        for date in data_dic:
            input_files = data_dic.get(date)
            output_file = output_path_merged + qua + "/" + date + ".tif"
            print("Merging for date:", date, qua, " files: ", input_files)

            if len(input_files) == 1:
                command = ['mv', input_files[0], output_file]
                subprocess.call(command)
                continue

            command = ['gdal_merge.py', '-init', str(no_data_value) , '-n', str(no_data_value), '-o', output_file] + input_files
            subprocess.call(command)

            for each_inp_file in input_files:
                os.remove(each_inp_file)

if __name__ == '__main__':
    download_lai_automatically()
    convert_h5_to_tif()
    chop_in_quadhash()
    merge_geotifs_for_quadhash()