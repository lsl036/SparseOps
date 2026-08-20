#!/usr/bin/env perl
# =======================================================================================
#
#      Filename:  generatePas.pl
#
#      Description:  Converter from ptt to pas file format.
#
#      Version:   <VERSION>
#      Released:  <DATE>
#
#      Author:  Jan Treibig (jt), jan.treibig@gmail.com
#      Project:  likwid
#
#      Copyright (C) 2016 RRZE, University Erlangen-Nuremberg
#
#      This program is free software: you can redistribute it and/or modify it under
#      the terms of the GNU General Public License as published by the Free Software
#      Foundation, either version 3 of the License, or (at your option) any later
#      version.
#
#      This program is distributed in the hope that it will be useful, but WITHOUT ANY
#      WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
#      PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
#      You should have received a copy of the GNU General Public License along with
#      this program.  If not, see <http://www.gnu.org/licenses/>.
#
# =======================================================================================

use lib 'util';
use strict;
use warnings;
use lib './perl';
use Template;
use Ptt;

my $PttInputPath = $ARGV[0];
my $PasOutputPath = $ARGV[1];
my $TemplateRoot = $ARGV[2];

my $tpl = Template->new({
        INCLUDE_PATH => ["$TemplateRoot"]
    });

$PttInputPath =~ /([A-Za-z_0-9]+)\.ptt/;
my $name = $1;

my ($PttVars, $TestcaseVars) = Ptt::ReadPtt($PttInputPath, $name);
$tpl->process('bench.tt', $PttVars, $PasOutputPath);
