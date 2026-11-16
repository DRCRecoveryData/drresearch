#!/usr/bin/perl -w
use strict;
use File::Basename;
use List::MoreUtils qw(uniq);

my $debug=1;

if(scalar(@ARGV)<3)
{
  print "Usage: $0 <input.dump> <output.dump> <pattern.xml> <casefile.case>\n";
  print "This tool searches the necessary parts from a dump to extract the parameters and writes it to the output.dump\n";
  print "Afterwards you can then upload the output.dump and send it to our reconstruction service\n";
  exit;
}


# TODO:
# TODO: Not all of the patterns are being found. This might be due to the patterns used being too strict
# TODO: We need a visualisation of which blocks were found and which blocks are missing
# TODO: We also need a visualisation in which blocks a pattern was found and in which blocks no patterns were found


my $imagefn=$ARGV[0];
my $dumpfn=$ARGV[1];
my $patternxmlfn=$ARGV[2];
my $casefn=$ARGV[3];

if(-f $dumpfn)
{
  print "Error: The output file $dumpfn already exists, and we want to avoid overwriting the wrong files. Please delete it and try again.\n";
  exit;
}

print "Extracting all relevant pages from a dump file \"$imagefn\" into an output dump \"$dumpfn\"\n";

my $pagesize=4000; # Bytes
my $eccstart=3145728;
my $eccend=3614367;
$pagesize=$1 if($ARGV[1]=~m/\((\d+)p\)/);
our $pagesperblock=128;
my $ecccoverage=1024;
my $blocksize=$pagesize*$pagesperblock; # !!! NEEDS TO BE ADAPTED LATER ON IN CASE THE VALUES CHANGED

my @datapos=(0,1500);
my $datasize=1024;
my @sapos=(3512);
my $sasize=8;
my @eccpos=(1024,2524);
my $eccsize=476;


if(open XML,"<$patternxmlfn")
{
  while(<XML>)
  {
    if(m/<pattern type='ECC' begin='(\d+)' end='(\d+)' size='\d+' coverage='(\d+)'/)
    {
      print "Loading pattern configuration from $ARGV[2]\n";
      $eccstart=$1;
      $eccend=$2;
      $ecccoverage=$3;
    }
  }
  close XML;
}


if(open CASE,"<$casefn")
{
  print "Reading from $casefn\n";
  my @mydatapos=();
  my @myeccpos=();
  my @mysapos=();
  while(<CASE>)
  {
    s/\x00//g; # FE is UTF-16
    $pagesize=$1 if(m/<Page_size>(\d+)<\/Page_size>/);
    $pagesize=$1 if(m/^Page +(\d+)\s*$/); # FE support
    if(m/^Block +0x([0-9a-fA-F]+)\s*$/) # FE support
    {
      print "FE Chip.txt format detected\n";
      $blocksize=hex($1);
      $pagesperblock=$blocksize/$pagesize;
    }
    if(m/<Actual_block_size>(\d+)<\/Actual_block_size>/)
    {
      $blocksize=$1;
      $pagesperblock=$blocksize/$pagesize;
    }
    if(m/<Record StructureDefinitionName="(DA|Data area|DATA)" StartAddress="(\d+)" StopAddress="(\d+)" \/>/i)
    {
      print "Adding $2 to datapos\n";
      push @mydatapos,$2;
      $datasize=$3-$2+1;
      $ecccoverage=$datasize;
    }
    if(m/<Record StructureDefinitionName="ECC" StartAddress="(\d+)" StopAddress="(\d+)" \/>/)
    {
      push @myeccpos,$1;
      $eccsize=$2-$1+1;
    }
    if(m/<Record StructureDefinitionName="SA" StartAddress="(\d+)" StopAddress="(\d+)" \/>/)
    {
      push @mysapos,$1;
      $sasize=$2-$1+1;
    }
    @datapos=uniq @mydatapos;
    @eccpos=uniq @myeccpos;
    @sapos=uniq @mysapos;
  }
  #$sectors=scalar(@datapos)*$datasize;
  close CASE;
}
else
{
  print STDERR "WARNING: There is no case file given as parameter, or it could not be opened: $!\nWARNING: We might be missing necessary geometry information. Please make sure that you are running the dumpextractrelevant tool with the casefile parameter.\n";
}


our $xorkey="";
my $keygiven=defined($ARGV[4]);
my $keyfn=$keygiven?$ARGV[4]:"$ARGV[0].xor";
if(open XOR,"<$keyfn")
{
  print "Loading XOR key from $keyfn\n";
  binmode XOR;
  undef $/;
  $xorkey=<XOR>;
  close XOR; 
}

print "Final configuration used:\nECC-start: Sector $eccstart\nECC-end: Sector $eccend\nECC-coverage: $ecccoverage Bytes\n";


open(IN,"<:raw",$imagefn) || die "Could not open dump file $imagefn for reading: $!\n";
binmode IN;
open(OUT,">:raw",$dumpfn) || die "Could not open dump file $dumpfn for writing: $!\n";
binmode OUT;

my $ende=0;
my $pagen=0;
my $outpages=0;

my $char1 = '|Block#';
my $char2 = "xxxxx\x0a\x00";


my %posfound=(); # This is an array of the offsets of the sector start positions inside a page
my %lbafound=();

# The record writes 'Byte: ' 33 bytes after its own start, so that literal still
# locates a sector whose '|Block#' magic has bit-rotted. Such a candidate is
# accepted only when the two decimal fields agree with each other, because
# nothing but the magic vouched for the offset. POSFOUND_LEGACY=1 restores the
# original behaviour: the hash is then never cleared, so a page is also scored at
# offsets marked by other pages, which picks up some rotted anchors by accident.
my $legacy=defined($ENV{POSFOUND_LEGACY}) && $ENV{POSFOUND_LEGACY}=~/^[1y]/i;
my $char3='Byte: ';
my $char4='(0x';
my $tolbits=defined($ENV{MAGTOL}) ? $ENV{MAGTOL} : 3;   # bits that may rot inside '|Block#'
my @pc = map { my $x=$_; my $c=0; $c++ while($x &=$x-1); $c } 0..255;
my @c1 = unpack('C7',$char1);
my @c2 = unpack('C7',$char2);

sub fixdecimal($) # Fixes up to 3 bit errors
{
  return $_[0] if(!defined($_[0]) || length($_[0])==0);
  my $str=$_[0];
  my $count=($str=~tr/[0-9]//);
  return $_[0] if($count==length($_[0]));
  if($count<length($_[0]) && $count>length($_[0])-3)
  {
    #print "Trying to fix $_[0]\n";
    my $d=$_[0];
    foreach(0 .. length($d)-1)
    {
      substr($d,$_,1)=pack("C",(unpack("C",substr($d,$_,1))&0xf|0x30));
    }
    return $d;
  }
  return $_[0];
}

#print fixdecimal("0123456789")."\n";
#print fixdecimal("012345v789")."\n";


print "Dump file size: ".(-s $imagefn)." (".((-s $imagefn)/$pagesize)." pages)\n";

if($debug)
{
  open BMAP,">$imagefn.blocks.map";
  open PMAP,">$imagefn.pages.map";
  print BMAP "# Mapfile. Created by dumprelevantextract\n# THIS DEBUG LOGFILE SHOWS THE LBA BLOCKS THAT WERE FOUND\n# Commandline. @ARGV\n# Start time: ".localtime(time())."\n0x0 + 1\n# pos      size   status\n";
  print PMAP "# Mapfile. Created by dumprelevantextract\n# THIS DEBUG LOGFILE SHOWS THE PAGES OF THE DUMP IN WHICH BLOCKS WERE FOND\n# Commandline. @ARGV\n# Start time: ".localtime(time())."\n0x0 + 1\n# pos      size   status\n";
}

my $prevpos=0;
my $prevval=42;

while(!$ende)
{
  my $in="";
  my $read=read IN,$in,$pagesize;
  last if(!defined($read) || !$read);

  if(0) #length($xorkey))
  {
    print "pagen%pagesperblock: ".($pagen % $pagesperblock)."\n";
    print "pagesperblock: $pagesperblock\n";
    print "pagesize: $pagesize\n";
    print "length xor: ".length($xorkey)."\n";
    print "start: ".($pagesize*($pagen % $pagesperblock))."\n";
    print "len part: ".length(substr($xorkey,$pagesize*($pagen % $pagesperblock),$pagesize))."\n";
  }

  if(length($xorkey))
  {
    # A one-block key is tiled over every block (the classic Phison case). A key
    # as long as the dump carries its own per-block/per-page keystream, so index
    # it absolutely and wrap only if the file is not a whole number of pages.
    my $koff=($pagesize*$pagen)%length($xorkey);
    my $k=substr($xorkey,$koff,$pagesize);
    $k.=substr($xorkey,0,$pagesize-length($k)) if(length($k)<$pagesize);
    $in^=$k;
  }
  my $sector=$in;
  %posfound=() if(!$legacy);

  my $offset=0;
  my $isgood=0;

  my $result = index($sector, $char1, $offset); # Search for the first sector inside this page by the pattern at the beginning of the sector, remember the startposition of the sector
  while ($result != -1)
  {
    $posfound{$result}=1;
    $offset = $result + 1; # Where to search for the next sector inside this page?
    $result = index($sector, $char1, $offset);
  }

  $offset=0;
  $result = index($sector, $char2, $offset); # Search for the first sector inside this page by the pattern at the end of the sector, remember the startposition of the sector
  while ($result != -1)
  {
    $posfound{$result-512+7}=1;
    $offset = $result + 1; # Where to search for the next sector inside this page?
    $result = index($sector, $char2, $offset);
  }

  if(!$legacy)
  {
    $offset=0;
    $result = index($sector, $char3, $offset); # Records whose magic rotted still spell 'Byte: ' 33 bytes in
    while ($result != -1)
    {
      my $start=$result-33;
      $posfound{$start}=3 if($start>=0 && !defined($posfound{$start}));
      $offset = $result + 1;
      $result = index($sector, $char3, $offset);
    }
    $offset=0;
    $result = index($sector, $char4, $offset); # '(0x' sits 20 bytes in, so it locates records whose 'Byte: ' rotted
    while ($result != -1)
    {
      my $start=$result-20;
      $posfound{$start}=4 if($start>=0 && !defined($posfound{$start}));
      $offset = $result + 1;
      $result = index($sector, $char4, $offset);
    }
    $offset=0;
    $result = index($sector, '|', $offset); # Three rotted bits can break every 2-byte run of '|Block#', so measure the rot wherever a '|' appears
    while ($result != -1)
    {
      $posfound{$result}=5 if(!defined($posfound{$result}));
      $offset = $result + 1;
      $result = index($sector, '|', $offset);
    }
    $offset=0;
    $result = index($sector, '#', $offset); # ... and the '|' itself can be the rotted byte, so anchor the last byte too
    while ($result != -1)
    {
      my $start=$result-6;
      $posfound{$start}=5 if($start>=0 && !defined($posfound{$start}));
      $offset = $result + 1;
      $result = index($sector, '#', $offset);
    }
  }

  foreach my $result (sort keys %posfound)
  {
    #print "Len: ".length($sector)." result:$result ".($result+59)."\n";
    next if(length($sector)<($result+59));
    my $lbah=substr($sector,$result+23,8);
    my $lbad=fixdecimal(substr($sector,$result+7,12));
    my $lbab=fixdecimal(substr($sector,$result+39,20));
    if(!defined($lbab))
    {
      print STDERR "WARNING: Most likely the pagesize is wrong. Please give the pagesize by naming the dump files like mydump(18324p).dmp\n";
      exit;
    }
    my $lba=undef;
    #print "Found $char at $result (fulladdress:$fulladdress xorpage:$xorpage blockpage:$blockpage)";
    my $lbaD=undef; $lbaD=int($lbad) if(defined($lbad) && $lbad=~m/^(\d+)$/);
    my $lbaH=undef; $lbaH=hex("0x".$1) if(defined($lbah) && $lbah=~m/^([0-9a-fA-F]+)$/);
    my $lbaB=undef; $lbaB=int($lbab/512) if(defined($lbab) && $lbab=~m/^(\d+)$/);
    # Majority-Voting on the LBA address
    $lba=$lbaD if(defined($lbaD) && defined($lbaH) && $lbaD == $lbaH);
    $lba=$lbaD if(defined($lbaD) && defined($lbaB) && $lbaD == $lbaB);
    $lba=$lbaH if(defined($lbaH) && defined($lbaB) && $lbaH == $lbaB);

    #print " LBA:$lba" if(defined($lba));
    #print " LBAd:$lbad($lbaD) LBAh:$lbah($lbaH) LBAb:$lbab($lbaB)" if(defined($lba));
    #print "\n";

    # Markers 3 and 4 come from a record literal, not from the magic, so the
    # record has to vouch for the position before its address may be counted.
    my $mode=$posfound{$result};

    if($mode>=3)
    {
      # How much of '|Block#' (or of the sector tail, which is the same record seen
      # from the other end) actually rotted.  Within $tolbits it is the author's own
      # exact-marker case, so the repaired fields are trusted as before; beyond that
      # the literals and the arithmetic have to carry it alone.
      my $rot=0; my $i=0;
      while($i<7 && $rot<=$tolbits) { $rot+=$pc[ord(substr($sector,$result+$i,1))^$c1[$i]]; $i++; }
      if($rot>$tolbits && $result+512<=length($sector))
      {
        my $trotd=0; $i=0;
        while($i<7 && $trotd<=$tolbits) { $trotd+=$pc[ord(substr($sector,$result+505+$i,1))^$c2[$i]]; $i++; }
        $rot=$trotd if($trotd<$rot);
      }
      if($rot>$tolbits)
      {
        # Random data gives two of '(0x' / ') ' / 'Byte: ' plus a clean 12- or 20-digit
        # field far less than once in a 42 GB dump, so this cannot fire by luck.
        my $lit=0;
        $lit++ if(substr($sector,$result+20,3) eq '(0x');
        $lit++ if(substr($sector,$result+31,2) eq ') ');
        $lit++ if(substr($sector,$result+33,6) eq 'Byte: ');
        next if($lit<2);
        my $cleandec=substr($sector,$result+7,12)=~m/^(\d{12})$/;
        my $cleanbyte=substr($sector,$result+39,20)=~m/^(\d{20})$/;
        my $agree=defined($lbaD) && defined($lbaB) && $lbaD == $lbaB;
        next if(!$agree && !$cleandec && !$cleanbyte);
        # Without the arithmetic check, a value only counts if its own field needed no repair.
        undef $lbaD if(!$cleandec && !$agree);
        undef $lbaB if(!$cleanbyte && !$agree);
      }
    }

    if(defined($lbaD) && $lbaD>=$eccstart && $lbaD<=$eccend)
    {
      $isgood=1;
      $lbafound{$lbaD}++;
    }
    if(defined($lbaB) && $lbaB>=$eccstart && $lbaB<=$eccend)
    {
      $isgood=1;
      $lbafound{$lbaB}++;
    }
    if(defined($lbaH) && $lbaH>=$eccstart && $lbaH<=$eccend)
    {
      $isgood=1;
      $lbafound{$lbaH}++;
    }
    #if($in=~m/1874827776/)
    #{
    #  print "Found 1874827776 D:$lbaD H:$lbaH B:$lbaB isgood:$isgood $eccstart $eccend\n" ;
    #  open DEB,">187.dat";
    #  binmode DEB;
    #  print DEB $in;
    #  close DEB;
    #}
  }
  #print "Found 1874827776\n" if($in=~m/1874827776/);
  $isgood=1 if(!$isgood && $in=~m/P00000/); # We want to populate the %lbafound first but still add the sector even if we dont find the phi pattern

  if($isgood)
  {
    print OUT $in;
    $outpages++;
    #print "Writing 1874827776\n" if($in=~m/1874827776/);
  }
  if($prevval ne $isgood)
  {
    print PMAP sprintf("0x%X 0x%X %s\n",$prevpos<<9,($prevpos-$pagen)<<9,$isgood?"+":"-") if($pagen);
    $prevval=$isgood;
    $prevpos=$pagen;
  }

  $pagen++;
  print STDERR "$pagen pages processed\n" if(!($pagen %100000));
}

close IN;
close OUT;

my $outsize=$outpages*$pagesize;
my $size=$pagen*$pagesize;
print "Input Image Size: $size Bytes ".($size/1000/1000/1000)." GB $pagen pages with pagesize $pagesize - $imagefn\n";
print "Output Dump Size: $outsize Bytes ".($outsize/1000/1000/1000)." GB $outpages Pages with pagesize $pagesize -> $dumpfn\n";

my $missing=0;
my $found=0;
my $inc=($ecccoverage/512)+1;
print "Inc: $inc\n";
my @missings=();
$prevpos=$eccstart;
$prevval=defined($lbafound{$eccstart});
for(my $lba=$eccstart; $lba<=$eccend; $lba+=$inc)
{
  $missing++ if(!defined($lbafound{$lba}));
  $found++ if(defined($lbafound{$lba}));
  if($prevval ne defined($lbafound{$lba}))
  {
    print BMAP sprintf("0x%X 0x%X %s\n",($prevpos-$eccstart)<<9,($lba-$prevpos+1)<<9,defined($lbafound{$lba})?"+":"-");
    $prevval=defined($lbafound{$lba});
    $prevpos=$lba+1;
  }
  push @missings,$lba if(!defined($lbafound{$lba}));
}
print BMAP sprintf("0x%X 0x%X %s\n",($eccend-$eccstart+1)<<9,($eccend-$prevpos+1)<<9,defined($prevval)?"+":"-");
close BMAP;
close PMAP;
# FOUNDLIST/MISSINGLIST: the grid points accepted/rejected, one LBA per line. Only
# wanted for comparing two versions' decisions; the counting above is unchanged.
for my $spec (["FOUNDLIST",1],["MISSINGLIST",0])
{
  my ($env,$want)=@$spec;
  next if(!$ENV{$env});
  open my $FH,">:raw",$ENV{$env} or die "Could not write $ENV{$env}: $!\n";
  for(my $lba=$eccstart; $lba<=$eccend; $lba+=$inc)
  {
    print $FH "$lba\n" if(defined($lbafound{$lba}) == $want);
  }
  close $FH;
}
my $percent=($found+$missing)?int(100*$found/($found+$missing)) : 0;
print "Found: $found Missing: $missing => $percent % found (Range searched: $eccstart..$eccend inc $inc)\n";
print "Missing: ".join(",",@missings)."\n" if($missing && $missing<20);
sub try_perblock($)
{
  my ($basekey)=@_;
  return 0 if($keygiven); # we are already running with an explicit key: no recursion
  my $pbkey="$imagefn.perblock.xor";
  my $solver=__FILE__; $solver=~s/\w+\.pl$/xorsolve_perblock.py/;
  print "\nxorsearch.pl assumes the same key in every block. Trying a key byte per block instead:\n\n";
  my $scmd="python \"$solver\" \"$imagefn\" \"$casefn\" \"$pbkey\"".($basekey?" --base \"$basekey\"":"");
  print "CMD: $scmd\n";
  system $scmd;
  return 0 if(!-f $pbkey);
  print "\nPer-block XOR key written to $pbkey, now trying again.\n\n";
  unlink $dumpfn;
  my $cmd="perl \"$0\" \"".join("\" \"",@ARGV)."\" \"$pbkey\"";
  print "Running $cmd\n";
  system $cmd;
  return 1;
}

if($percent<99)
{
  print "\nThis dump is incomplete or has not been properly XOR decoded yet.\n";
  if(-f "$imagefn.xor")
  {
    if(try_perblock("$imagefn.xor")) { exit; }
    print "Perhaps the XOR key is wrong? We give up now.\n";
    print "A key that changes from block to block is worth a dump-sized key file, not a one-block key:\n";
    print "  python xorsolve_perblock.py \"$imagefn\" \"$casefn\" \"$imagefn.perblock.xor\" --base \"$imagefn.xor\"\n";
    print "then: perl $0 \"".join("\" \"",@ARGV)."\" \"$imagefn.perblock.xor\"\n";
    exit;
  }
  if(!defined($casefn))
  {
    print STDERR "Error: Without the geometry information from a casefile we cannot recover the XOR key!\n";
    exit;
  }
  print "Trying to automatically XOR decode it now:\n\n";
  my $xorsearch=__FILE__; $xorsearch=~s/\w+\.pl$/xorsearch.pl/;
  my $cmd="perl \"$xorsearch\" \"$imagefn\" \"$imagefn.xor\" \"$casefn\"";
  print "CMD: $cmd\n";
  system $cmd;
  if(-f "$imagefn.xor")
  {
    print "\nXOR key was found, now trying again.\n\n";
    unlink $dumpfn;
    my $cmd="perl \"$0\" \"".join("\" \"",@ARGV)."\" ";
    print "Running $cmd\n";
    system $cmd;
  }
  else
  {
    if(try_perblock("")) { exit; }
    print "\nThe XOR key could not be found automatically, please check whether the pattern has been written to the disk/card correctly, and whether the geometry of the dump is correct\n";
    print "Please check the size and make sure it has been XOR decoded properly. If that doesn't help, please provide the whole dump.\n";
  }
}

#print "Page Offsets for DATA: ".join(",",sort keys(%posfound))."\n";
print STDERR "Done.\n";
