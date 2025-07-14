my $Press = 3000;
for (my $i = 300; $i >= 160; $i = $i - 5) {
	for (my $j = 1; $j <=1; $j++) {
		my $filename1 = 'P'.$Press.'_T'.$i.'/data'.$j;
		system("cd ./Linear_0.2/slope_455 && python3 dp_LDL_simple_fhi47_100_linear_test_auto.py $filename1 3");
		#system("mv /home/dammerung/workstation/train/ALL_Map_PT_ulta_smallNewtwork/Linear_0.2/slope_455/logtest/xe.log /home/dammerung/workstation/train/ALL_Map_PT_ulta_smallNewtwork/Linear_0.2/slope_455/logtest/xe$j.log");
		system("cd ./Linear_0.4/slope_490 && python3 dp_LDL_simple_fhi47_100_linear_test_auto.py $filename1 3");
		#system("mv /home/dammerung/workstation/train/ALL_Map_PT_ulta_smallNewtwork/Linear_0.4/slope_490/logtest/xe.log /home/dammerung/workstation/train/ALL_Map_PT_ulta_smallNewtwork/Linear_0.4/slope_490/logtest/xe$j.log");
	}
	system("cd ./Linear_0.2/slope_455/logtest && perl read.pl");
	system("cd ./Linear_0.4/slope_490/logtest && perl read.pl");
	my $filename = '_'.$Press.'_'.$i;
	system("mv ./Linear_0.2/slope_455/logtest/xe.log ./Linear_0.2/slope_455/logtest/xe$filename.log");
	#system("mv /home/dammerung/workstation/train/ALL_Map_PT_ulta_smallNewtwork/Linear_0.2/slope_455/logtest/xe2.log /home/dammerung/workstation/train/ALL_Map_PT_ulta_smallNewtwork/Linear_0.2/slope_455/logtest/xe2$filename.log");
	system("mv ./Linear_0.2/slope_455/logtest/xe.txt ./Linear_0.2/slope_455/logtest/xe$filename.txt");
	##
	system("mv ./Linear_0.4/slope_490/logtest/xe.log ./Linear_0.4/slope_490/logtest/xe$filename.log");
	#system("mv /home/dammerung/workstation/train/ALL_Map_PT_ulta_smallNewtwork/Linear_0.4/slope_490/logtest/xe2.log /home/dammerung/workstation/train/ALL_Map_PT_ulta_smallNewtwork/Linear_0.4/slope_490/logtest/xe2$filename.log");
        system("mv ./Linear_0.4/slope_490/logtest/xe.txt ./Linear_0.4/slope_490/logtest/xe$filename.txt");
	print("=============================\n");
	#my $filename2 = 'P'.$Press.'_T'.$i.'/data2';
}
