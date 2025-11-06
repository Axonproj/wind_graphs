#!/usr/bin/env perl
use strict;
use warnings;
use feature 'say';
use File::Spec;
use POSIX qw(strftime);
use Time::Piece;
use Time::Seconds;

# --- DATE HANDLING ---------------------------------------------------------
# Optional command-line arg: yyyymmdd (treated as "today")
my $today_str;

if (@ARGV && $ARGV[0] =~ /^\d{8}$/) {
    $today_str = $ARGV[0];
    say "[info] Using provided date as 'today': $today_str";
} else {
    $today_str = strftime("%Y%m%d", localtime);
    say "[info] Using system date as 'today': $today_str";
}

# Compute yesterday and tomorrow
my $today_obj = Time::Piece->strptime($today_str, "%Y%m%d");
my $yesterday = ($today_obj - ONE_DAY)->strftime("%Y%m%d");
my $tomorrow  = ($today_obj + ONE_DAY)->strftime("%Y%m%d");

say "[info] Today: $today_str  Yesterday: $yesterday  Tomorrow: $tomorrow";

# --- EXISTING PIPELINE LOGIC ---
# (Your existing data download and processing code stays here)

# --- GRAPH GENERATION -----------------------------------------------------
my $graph_file = File::Spec->catfile("graphs", "${yesterday}_bramble_bank_wind.png");

if (-e $graph_file) {
    say "[skip] Graph already exists: $graph_file";
} else {
    say "[run] Generating graph using make_graph.py: $graph_file";
    my $cmd = "python3 make_graph.py $yesterday --no-show";
    my $exit_status = system($cmd);

    if ($exit_status == 0) {
        say "[ok] Graph created successfully: $graph_file";

        # --- UPDATE index.html -----------------------------------------------------
        my $index_file = "index.html";

        if (-e $index_file) {
            say "[update] Adding new graph to index.html: $graph_file";

            open my $in, "<", $index_file or die "Cannot read $index_file: $!";
            my @lines = <$in>;
            close $in;

            my $img_tag = qq{  <img src="$graph_file"\n       style="display:block; margin:auto; max-width:100%; height:auto;">\n};

            open my $out, ">", $index_file or die "Cannot write to $index_file: $!";
            my $inserted = 0;
            foreach my $line (@lines) {
                print $out $line;
                if ($line =~ m{<h1>} && !$inserted) {
                    print $out $img_tag;
                    $inserted = 1;
                }
            }
            close $out;

            say "[ok] Updated index.html with new image.";
        } else {
            warn "[warn] index.html not found, skipping update.";
        }

    } else {
        warn "[warn] make_graph.py failed (exit code $exit_status)";
    }
}
