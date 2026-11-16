package List::MoreUtils;
use strict; use warnings; use Exporter;
our @ISA = qw(Exporter); our @EXPORT_OK = qw(uniq); our @EXPORT = qw(uniq);
sub uniq { my %s; grep { !$s{$_}++ } @_; }
1;
