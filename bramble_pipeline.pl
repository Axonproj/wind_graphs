#!/usr/bin/env perl
use strict;
use warnings;
use Time::Piece;
use Time::Seconds;
use File::Path qw(make_path);
use File::Spec;
use HTTP::Tiny;
use Text::CSV;

# --- Config
my $DATA_DIR        = "data";
my $FORECAST_SCRIPT = "./get_forecast.sh";

sub say { print @_, "\n" }

# --- Dates (yyyyMMdd)
my $now       = localtime;
my $today     = $now->strftime('%Y%m%d');
my $y_t       = $now - ONE_DAY;
my $yesterday = $y_t->strftime('%Y%m%d');
my $t_t       = $now + ONE_DAY;
my $tomorrow  = $t_t->strftime('%Y%m%d');

say "[info] Today:     $today";
say "[info] Yesterday: $yesterday";
say "[info] Tomorrow:  $tomorrow";

# --- Ensure data dir
unless (-d $DATA_DIR) {
    say "[info] Creating data directory: $DATA_DIR";
    make_path($DATA_DIR) or die "[error] Could not create '$DATA_DIR': $!\n";
}

# --- Build URL & output paths for YESTERDAY
my $yyyy         = $y_t->strftime('%Y');
my $day_of_month = $y_t->strftime('%d'); # 01..31

# Use fixed English month names/abbreviations (avoid locale issues)
my @month_names = qw(January February March April May June July August September October November December);
my @month_abbr  = qw(Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec);
my $mon_index   = $y_t->mon - 1; # 0..11

my $month_name   = $month_names[$mon_index];
my $month_abbrev = $month_abbr[$mon_index];

# Example: https://www.bramblemet.co.uk/archive/2025/January/CSV/Bra01Jan2025.csv
my $source_csv_url = sprintf(
    "https://www.bramblemet.co.uk/archive/%s/%s/CSV/Bra%s%s%s.csv",
    $yyyy, $month_name, $day_of_month, $month_abbrev, $yyyy
);

my $actual_csv    = File::Spec->catfile($DATA_DIR, "${yesterday}_bramble_actual.csv");
my $forecast_json = File::Spec->catfile($DATA_DIR, "${tomorrow}_bramble_forecast.json");

# --- YESTERDAY: fetch actual wind data if missing (no curl; pure Perl HTTP)
if (-e $actual_csv) {
    say "[ok] Actual file already exists: $actual_csv";
} else {
    say "[info] Actual file not found. Fetching yesterday's data for $yesterday …";
    say "[info] URL: $source_csv_url";

    my $http = HTTP::Tiny->new( timeout => 20 );
    my $res  = $http->get($source_csv_url);

    unless ($res->{success}) {
        my $status = $res->{status} // '???';
        my $reason = $res->{reason} // 'Unknown error';
        die "[error] HTTP GET failed ($status): $reason\n";
    }

    my $content = $res->{content};
    open my $out, ">", $actual_csv
        or die "[error] Cannot write '$actual_csv': $!\n";

    my $csv = Text::CSV->new({ binary => 1, auto_diag => 1, allow_loose_quotes => 1 });

    my $rows = 0;
    eval {
        open my $in, "<", \$content or die "Cannot open in-memory: $!";
        while (my $line = <$in>) {
            next if $line =~ /^\s*$/; # skip blanks
            if ($csv->parse($line)) {
                my @f = $csv->fields;
                # Keep columns 1,2,3,5 => indexes 0,1,2,4
                my @want = map { defined $f[$_] ? $f[$_] : "" } (0,1,2,4);
                print $out join(",", @want), "\n";
                $rows++;
            }
        }
        close $in;
        1;
    } or do {
        my $err = $@ || "unknown error";
        close $out;
        unlink $actual_csv; # remove partial file
        die "[error] Failed while parsing CSV: $err";
    };

    close $out;

    if ($rows > 0) {
        say "[ok] Saved actual data ($rows rows) to: $actual_csv";
    } else {
        unlink $actual_csv;
        warn "[error] No rows parsed; removed empty file: $actual_csv\n";
    }
}

# --- TOMORROW: get forecast if missing
if (-e $forecast_json) {
    say "[ok] Forecast file already exists: $forecast_json";
} else {
    say "[info] Forecast file not found. Running forecast script: $FORECAST_SCRIPT";
    unless (-x $FORECAST_SCRIPT) {
        die "[error] Forecast script not found or not executable: $FORECAST_SCRIPT\n";
    }
    my $rc = system($FORECAST_SCRIPT);
    if ($rc == 0) {
        if (-e $forecast_json) {
            say "[ok] Forecast saved to: $forecast_json";
        } else {
            warn "[warn] Forecast script completed but $forecast_json was not created.\n";
        }
    } else {
        my $exit = ($rc == -1) ? "failed to execute: $!" : "exit status " . ($rc >> 8);
        warn "[error] Forecast script failed ($exit). Forecast not updated.\n";
    }
}

say "[done] Finished. Check '$DATA_DIR' for outputs.";
