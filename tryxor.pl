#!/usr/bin/perl -w

sub readfile($)
{
  if(open(RFIN,"<$_[0]"))
  {
    my $old=$/;
    undef $/;
    binmode RFIN;
    my $content=<RFIN>;
    $/=$old;
    close RFIN;
    return($content);
  }
  return "";
}


my $fndump=$ARGV[0];
my $fnxor=$ARGV[1];
my $fnxorp=sprintf("%74s",$fnxor);

#print "Loading dump $fndump and XOR file $fnxor\n";

my $xor=readfile($fnxor);
my $xorl=length($xor);

open IN,"<$fndump";
binmode IN;
binmode STDOUT;

my $total=0;
my $found=0;

my $sector="";
my $wanted=$xorl/10;
while($total<10000000000)
{
  read IN,$sector,$xorl;
  #print STDERR "LEN: ".length($sector)."\n";

  my $decoded=$sector ^ $xor;
  #print STDERR "DECODED: ".length($sector)."\n";

  my $count =0; #($decoded =~ tr/x//);
  $count++ if($decoded=~m/Block/);
  print $decoded if($decoded=~m/Block/);

  #print STDERR "$count\n";

  $found+=$count;

  $total+=$xorl;
  #last;
}

print STDERR "$fnxorp XORlen: $xorl modulo18336: ".($xorl%(18336))." Total: $total   Found: $found\n";

