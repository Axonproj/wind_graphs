#!/usr/bin/env perl
# fetch_openmeteo_forecast.pl yyyymmdd
#
# Saves the AROME HD (Meteo-France, 1.5 km) and ECMWF IFS (9 km) wind
# forecasts for Bramble Bank, in knots, from the free Open-Meteo API to
# data/{yyyymmdd}_bramble_openmeteo.json, for comparison with the Met Office
# forecast and the Bramblemet actuals.
use strict;
use warnings;
use Time::Piece;
use Time::Seconds;
use HTTP::Tiny;

my $DATA_DIR = "data";
my $MODELS   = "meteofrance_arome_france_hd,ecmwf_ifs";
my ($LAT, $LON) = (50.78, -1.28);    # same point as the Met Office request

my $date = shift // die "Usage: $0 yyyymmdd\n";
$date =~ /^\d{8}$/ or die "[error] Date must be yyyymmdd, got '$date'\n";

# Times are UTC; the London day starts at 23:00 UTC the day before in summer
my $day   = Time::Piece->strptime($date, '%Y%m%d');
my $start = ($day - ONE_DAY)->strftime('%Y-%m-%d');
my $end   = $day->strftime('%Y-%m-%d');

my $url = "https://api.open-meteo.com/v1/forecast"
        . "?latitude=$LAT&longitude=$LON"
        . "&hourly=wind_speed_10m,wind_gusts_10m,wind_direction_10m"
        . "&wind_speed_unit=kn&timezone=GMT"
        . "&start_date=$start&end_date=$end&models=$MODELS";

my $res = HTTP::Tiny->new( timeout => 30 )->get($url);
unless ($res->{success}) {
    die "[error] Open-Meteo GET failed ($res->{status}): $res->{reason}\n$res->{content}\n";
}

my $out_file = "$DATA_DIR/${date}_bramble_openmeteo.json";
open my $out, ">", $out_file or die "[error] Cannot write '$out_file': $!\n";
print $out $res->{content};
close $out;

print "[ok] Open-Meteo forecast saved to: $out_file\n";
