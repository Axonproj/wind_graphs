# 2023 (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of Weather DataHub and is released under the
# BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
# (c) Met Office 2023

use strict;
use warnings;
use LWP::UserAgent;
use URI;
use Getopt::Long;
use Time::HiRes qw(sleep);

my $base_url = "https://data.hub.api.metoffice.gov.uk/sitespecific/v0/point/";

# Default values
my $timesteps = 'hourly';
my $exclude_metadata = 'FALSE';
my $include_location = 'TRUE';
my $latitude = '';
my $longitude = '';
my $apikey = '';

# Parse command line arguments
GetOptions(
    't|timesteps=s'      => \$timesteps,
    'm|metadata=s'       => \$exclude_metadata,
    'n|name=s'           => \$include_location,
    'y|latitude=s'       => \$latitude,
    'x|longitude=s'      => \$longitude,
    'k|apikey=s'         => \$apikey,
) or die "Error in command line arguments\n";

# Open log file
open(my $log_fh, '>', 'ss_download.log') or die "Cannot open log file: $!\n";

sub log_message {
    my ($level, $message) = @_;
    my $timestamp = localtime();
    print $log_fh "$timestamp - $level - $message\n";
}

sub retrieve_forecast {
    my ($base_url, $timesteps, $apikey, $latitude, $longitude, $exclude_metadata, $include_location) = @_;
    
    my $url = $base_url . $timesteps;
    
    my $ua = LWP::UserAgent->new();
    $ua->timeout(30);
    
    my $uri = URI->new($url);
    $uri->query_form(
        excludeParameterMetadata => $exclude_metadata,
        includeLocationName => $include_location,
        latitude => $latitude,
        longitude => $longitude
    );
    
    my $success = 0;
    my $retries = 5;
    my $response;
    
    while (!$success && $retries > 0) {
        eval {
            $response = $ua->get(
                $uri,
                'Accept' => 'application/json',
                'apikey' => $apikey
            );
            $success = 1;
        };
        
        if ($@) {
            log_message('WARNING', "Exception occurred: $@");
            $retries--;
            sleep(10);
            
            if ($retries == 0) {
                log_message('ERROR', "Retries exceeded");
                close($log_fh);
                exit(1);
            }
        }
    }
    
    if ($response->is_success) {
        print $response->decoded_content;
    } else {
        log_message('ERROR', "HTTP request failed: " . $response->status_line);
        close($log_fh);
        exit(1);
    }
}

# Validation
if ($apikey eq '') {
    print "ERROR: API credentials must be supplied.\n";
    close($log_fh);
    exit(1);
}

if ($latitude eq '' || $longitude eq '') {
    print "ERROR: Latitude and longitude must be supplied\n";
    close($log_fh);
    exit(1);
}

if ($timesteps ne 'hourly' && $timesteps ne 'three-hourly' && $timesteps ne 'daily') {
    print "ERROR: The available frequencies for timesteps are hourly, three-hourly or daily.\n";
    close($log_fh);
    exit(1);
}

# Retrieve the forecast
retrieve_forecast($base_url, $timesteps, $apikey, $latitude, $longitude, $exclude_metadata, $include_location);

close($log_fh);
