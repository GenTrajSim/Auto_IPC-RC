open IN1, "< xe1.log";
open IN2, "< xe2.log";
open IN3, "< xe3.log";
open OUT, "+> xe.txt";
my @in1 = ();
while (<IN1>) {push @in1, [split];}
my $i = 0;
foreach my $a(@in1){
	if (($i%309)<=8) {
	}else{
		my $local_pot = $a -> [5];
		my $local_rho = $a -> [6];
		my $local_rc2 = $a -> [7];
		$local_rc2 = (split(/\]/,$local_rc2))[0];
		print OUT "$local_pot $local_rho $local_rc2\n";
	}
	$i++;
}
close IN1;
##
my @in2 = ();
while (<IN2>) {push @in2, [split];}
$i = 0;
foreach my $a(@in2){
        if (($i%309)<=8) {
        }else{
                my $local_pot = $a -> [5];
                my $local_rho = $a -> [6];
                my $local_rc2 = $a -> [7];
                $local_rc2 = (split(/\]/,$local_rc2))[0];
                print OUT "$local_pot $local_rho $local_rc2\n";
        }
        $i++;
}
close IN2;
##
my @in3 = ();
while (<IN3>) {push @in3, [split];}
$i = 0;
foreach my $a(@in3){
        if (($i%309)<=8) {
        }else{
                my $local_pot = $a -> [5];
                my $local_rho = $a -> [6];
                my $local_rc2 = $a -> [7];
                $local_rc2 = (split(/\]/,$local_rc2))[0];
                print OUT "$local_pot $local_rho $local_rc2\n";
        }
        $i++;
}
close IN3;
##
close OUT;
