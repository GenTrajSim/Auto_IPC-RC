use Getopt::Long;
$PT = 0;
GetOptions (
        'PT=s'=>\$PT,
);
print "$PT\n";
open IN, "< xe$PT.log";
open OUT, "+> xe$PT.txt";
my @in = ();
while (<IN>) {push @in, [split];}
my $i = 0;
foreach my $a(@in){
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
close IN;
close OUT;
