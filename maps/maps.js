/*
 * I've been eating a lot of noodles lately.
 * This is what I call "stream of consciousness" code to be
 * refactored later.
 */

if(typeof(String.prototype.trim) === "undefined") {
    String.prototype.trim = function() {
        return String(this).replace(/^\s+|\s+$/g, '');
    };
}

var descriptions = {};
var tags = {};

var entries = [];
var chosen = [];

function chart(tagtype) {
    $('#plotviewchart').html('');
    var tagdict = tags[tagtype]
    var uniques = tagdict.uniques;
    var data = []
    var sumcount = 0

    for (u in uniques) {
        count = 0

        for (e in chosen) {
            if (has_all_tags([uniques[u]], [tagdict.catnames[0]], chosen[e].data)) {
                 count += 1
            }
        }

        sumcount += count;

        datum = []
        datum.push(uniques[u])
        datum.push(count)
        data.push(datum)
    }

    for (d in data) {
        data[d].push(data[d][1] / sumcount)
    }

    console.log(data)

    var chart = d3.select("#plotviewchart");
    console.log(chart)

    chart.selectAll("div")
        .data(data)
        .enter().append("div")
            .style(
                "height",
                function(d) { return (100.0 / data.length) + "%"; }
            )
            .html(function(d) { return '<span class="name">' + d[0] + '</span>' })
            .append("div")
                .style(
                    "width",
                    function(d) { return d[2] * 100 + "%"; }
                )
                .style(
                    "line-height",
                    function(d) { return "100%"; }
                )
                .html(function(d) { return "<span class='count'>" + d[1] + "</span>"; });
}

function add_unique_tags(datum) {
    // Collect all the unique tags.
    for (tag in tags) {
        tagdict = tags[tag];

        for (var i = 0; i < tagdict.catnames.length; i++) {
            catname = tagdict.catnames[i];

            if (catname in datum) {
                // If element is not a list, make it one.
                elements = datum[catname];

                if (typeof(elements) == 'string')
                    elements = [elements];

                for (var j = 0; j < elements.length; j++) {
                    element = elements[j];

                    // Split the individual tag if necessary.
                    tokens = [element];

                    if ('split' in tagdict) {
                        tokens = element.split(tagdict['split']);
                    }

                    // Maximum number of split items to consider
                    // tags.
                    ml = tokens.length;

                    if ('splitlevel' in tagdict) {
                        ml = Math.min(
                            tokens.length,
                            tagdict.splitlevel
                        )
                    }

                    // Add tokens if they aren't already in the list.
                    for (var k = 0; k < ml; k++) {
                        var token = tokens[k];

                        if (tagdict.uniques.indexOf(token) < 0) {
                            tagdict.uniques.push(token.trim())
                        }
                    }
                }
            }
        }
    }
}

function create_entry_citation(datum) {
    var citation = $('<p></p>');

    if ('authors' in datum) {
        var authors = $('<span class="authors"></span>');
        text = '';

        for (var i = 0; i < datum.authors.length; i++) {
            text += datum.authors[i];

            if (i < datum.authors.length - 2)
                text += ', '
            else if (i < datum.authors.length - 1)
                text += ', and '
        }

        authors.text(text);
        citation.append(authors);

        if ('name' in datum || 'url' in datum || 'year' in datum)
            citation.append(', ')
    }

    if ('name' in datum || 'url' in datum) {
        var mapname = $('<a class="name"></a>');

        if (!('name' in datum))
            mapname.text(datum.url);
        else
            mapname.text(datum.name);

        if ('url' in datum)
            mapname.attr('href', datum.url);

        citation.append(mapname);

        if ('year' in datum)
            citation.append(', ')
    }

    if ('year' in datum) {
        year = $('<span class="year"></span>');
        year.text(datum.year);
        citation.append(year);
    }

    citation.append('.')

    return citation;
}

function create_entry_tags(datum) {
    var entrytags = $('<ul></ul>');

    for (tag in tags) {
        tagdict = tags[tag];

        if (tagdict.listtags) {
            for (var i = 0; i < tagdict.catnames.length; i++) {
                catname = tagdict.catnames[i];


                if (catname in datum) {
                    category = datum[catname];

                    if (typeof(datum[catname]) == 'string') {
                        category = [category];
                    }

                    for (var j = 0; j < category.length; j++) {
                        tagli = $('<li></li>');
                        tagli.addClass(tag);
                        tagli.text(category[j]);
                        entrytags.append(tagli);
                    }
                }
            }
        }
    }

    return entrytags;
}

function create_entry_element(datum) {
    var entryelement = $('<li></li>');
    entryelement.append(create_entry_citation(datum));
    entryelement.append(create_entry_tags(datum));

    return entryelement;
}

function add_selectable_tags() {
    for (tag in tags) {
        tagdict = tags[tag];

        for (var i = 0; i < tagdict.uniques.length; i++) {
            plottagli = $('<li></li>');
            plottagli.addClass(tag);
            plottagli.text(tagdict.uniques[i]);
            $('.plottags.' + tag).append(plottagli)

            tagli = $('<li></li>');
            tagli.addClass(tag);
            tagli.text(tagdict.uniques[i]);

            tagli.click(function() {
                ele = $(this);

                if (ele.hasClass('selected'))
                    ele.removeClass('selected');
                else
                    ele.addClass('selected');

                filter_entries();
            });

            tagli.mouseenter(function() {
                ttext = $(this).text().trim();

                if (ttext in descriptions) {
                    $('#hoverboard').text(descriptions[ttext]);

                    $('#hoverboard').css({
                        left: $('nav').offset().left - $('#hoverboard').width() - 56,
                        top: $(this).offset().top - $('#hoverboard').height() / 2
                    });

                    $('#hoverboard').show();
                }
            });

            tagli.mouseleave(function() {
                $('#hoverboard').hide();
            });

            $('#' + tag).append(tagli);
        }
    }
}

function has_all_tags(selected_tags, catnames, datum) {
    if (selected_tags.length) {
        for (var i = 0; i < selected_tags.length; i++) {
            var stag = selected_tags[i];
            var found = false;

            for (var j = 0; j < catnames.length; j++) {
                var cat = catnames[j];

                if (cat in datum) {
                    if (datum[cat].indexOf(stag) >= 0) {
                        found = true;
                        break;
                    }
                }
            }

            if (!found)
                return false;
        }

        return true;
    } else
        return true;
}

function has_any_tags(selected_tags, catnames, datum) {
    for (var i = 0; i < catnames.length; i++) {
        var cat = catnames[i];

        if (cat in datum) {
            for (var j = 0; j < selected_tags.length; j++) {
                stag = selected_tags[j];

                if (datum[cat].indexOf(stag) >= 0)
                    return true;
            }
        }
    }

    return false;
}

function filter_entries() {
    $('#selected').html('');
    $('#remaining').html('');
    chosen = []

    var selected_tags = {};

    for (var tag in tags) {
        selected_tags[tag] = [];

        $('#' + tag).children().each(function() {
            if ($(this).hasClass('selected')) {
                selected_tags[tag].push($(this).text().trim());
            }
        })
    }

    for (var i = 0; i < entries.length; i++) {
        var entry = entries[i];

        var included = true;

        for (tag in tags) {
            tagdict = tags[tag];
            stags = selected_tags[tag];

            if (stags.length) {
                if (tagdict.querytype == 'and') {
                    if (!has_all_tags(stags, tagdict.catnames, entry.data)) {
                        included = false;
                        break;
                    }
                } else {
                    if (!has_any_tags(stags, tagdict.catnames, entry.data)) {
                        included = false;
                        break;
                    }
                }
            }
        }

        if (included) {
            $('#selected').append(entry.element);
            chosen.push(entry);
        } else
            $('#remaining').append(entry.element);
    }

    $('#selectedtitle').text(
        'Selected maps (' + $('#selected').children().length + '/' + entries.length + ')'
    );

    $('#remainingtitle').text(
        'Remaining maps (' + $('#remaining').children().length + '/' + entries.length + ')'
    );
}

$(document).ready(function() {
    $('#hoverboard').hide();
    $('#plotview').hide();

    $('#plotfeatures').click(function() {chart('features')});
    $('#plotmedia').click(function() {chart('media')});
    $('#plotcontent').click(function() {chart('content')});
    $('#plotregion').click(function() {chart('regions')});
    $('#plotyear').click(function() {chart('years')});

    $('a#plot').click(function() {
        $('#plotview').fadeIn(500);
    });

    $('#plotview a.close').click(function() {
        $('#plotview').fadeOut(500);
    });

    $.get('config.json', function(data) {
        descriptions = data.descriptions;
        tags = data.tags;

        for (var tag in tags) {
            tags[tag]['count'] = 0;
            tags[tag]['uniques'] = [];
        }

        $.get('maps.json', function(data) {
            for (var i = 0; i < data.length; i++) {
                var datum = data[i];

                add_unique_tags(datum);
                element = create_entry_element(datum);

                // Add the entry.
                entry = {
                    data: data[i],
                    element: create_entry_element(datum)
                };

                entries.push(entry);
            }

            entries.sort(function(ea, eb) {
                aa = ea.data['authors'][0];
                ab = eb.data['authors'][0];

                if (aa < ab)
                    return -1;
                else if (aa > ab)
                    return 1;
                else
                    return 0;
            })

            for (tag in tags) {
                tags[tag].uniques.sort();
            }

            add_selectable_tags();
            filter_entries();
        });
    });
});
