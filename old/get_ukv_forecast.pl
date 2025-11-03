use strict;
use warnings;
use feature 'say';
use JSON qw(decode_json);
use LWP::UserAgent;
use URI::Escape;

# --- Configuration ---
my $API_KEY     = $ENV{METOFFICE_API_KEY} // die "Set METOFFICE_API_KEY in env\n";
my $LAT         = 50.78;     # Bramble Bank (approx)
my $LON         = -1.31;
my $RUN_TIME    = '2025-10-22T16:00:00Z';  # model run
my $DURATION_HR = 24;                      # get 24h forecast
my $PARAMS      = 'windSpeed10m,windGust10m';  # parameters to fetch

# --- API Endpoint ---
my $BASE_URL = 'https://api-metoffice.apiconnect.ibmcloud.com/metoffice/production/v0/sitespecific/v0/point/hourly';
my $url = "$BASE_URL?" .
    "latitude=$LAT&longitude=$LON" .
    "&modelRunDate=" . uri_escape($RUN_TIME) .
    "&duration=$DURATION_HR" .
    "&parameters=$PARAMS";

# --- HTTP request ---
my $ua = LWP::UserAgent->new(timeout => 20);
$ua->default_header('x-ibm-client-id' => $API_KEY);
$ua->default_header('accept' => 'application/json');

say "Fetching forecast from UKV model...";
my $response = $ua->get($url);

unless ($response->is_success) {
    die "HTTP Error: " . $response->status_line . "\n" . $response->decoded_content;
}

# --- Parse JSON ---
my $data = decode_json($response->decoded_content);

# --- Extract and print times + wind speeds ---
say "Time (UTC), WindSpeed10m (m/s), WindGust10m (m/s)";
for my $t (@{ $data->{features}[0]{properties}{timeSeries} }) {
    my $time  = $t->{time};
    my $speed = $t->{windSpeed10m} // 'NA';
    my $gust  = $t->{windGust10m} // 'NA';
    say "$time, $speed, $gust";
}
