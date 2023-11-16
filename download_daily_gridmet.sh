if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <year>"
    exit 1
fi

year="$1"
base_url="https://www.northwestknowledge.net/metdata/data/"

wget -nc -c -nd "${base_url}vpd_${year}.nc"
wget -nc -c -nd "${base_url}etr_${year}.nc"
wget -nc -c -nd "${base_url}pet_${year}.nc"
wget -nc -c -nd "${base_url}srad_${year}.nc"
wget -nc -c -nd "${base_url}tmmn_${year}.nc"
wget -nc -c -nd "${base_url}tmmx_${year}.nc"
wget -nc -c -nd "${base_url}rmax_${year}.nc"
wget -nc -c -nd "${base_url}rmin_${year}.nc"
wget -nc -c -nd "${base_url}pr_${year}.nc"




