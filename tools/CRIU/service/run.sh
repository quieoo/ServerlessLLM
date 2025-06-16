
function title_print {
	echo -e "\n**************************************************"
	echo -e "\t\t"$1
	echo -e "**************************************************\n"

}

rm -rf imgs_py_2
mkdir -p imgs_py_2

title_print "Run test_2.py"
setsid python test_2.py criu_service.socket imgs_py_2 < /dev/null &>> output_py_2

sleep 1  # 新增等待

title_print "Restore test_2.py"
setsid criu restore -v4 -o restore-py_2.log -D imgs_py_2  < /dev/null &>> output_py_2