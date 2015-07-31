var cscale = 20;

function get_pixel_channel(x, y, channel) {
  var d = window.devicePixelRatio * cscale;
  return pixels[4 * ((y * d) * (width * d) + (x * d)) + channel];
}

function set_pixel_channel(x, y, value, channel) {
  var d = window.devicePixelRatio * cscale;

  for (var i = 0; i < d; i++) {
    for (var j = 0; j < d; j++) {
      idx = 4 * ((y * d + i) * (width * d) + (x * d + j));
      pixels[idx + channel] = value;
    }
  }
}

function setup() {

  console.log(cscale);
  var canvas = createCanvas(5 * cscale, 32 * cscale);
  console.log(width, height);
  canvas.class("p5c");
  canvas.parent("p5");

  clear();
  loadPixels();

  for (var i = 0; i < height / cscale; i++) {
    for (var j = 0; j < width / cscale; j++) {
      for (var channel = 0; channel < 4; channel++) {
        set_pixel_channel(j, i, (128 + i + j) % 256, channel);
      }
    }
  }

  updatePixels();
}

function draw() {
  loadPixels();

  for (var channel = 0; channel < 4; channel ++) {
    for (var row = 0; row < height / cscale; row++) {
      for (var col = 0; col < width / cscale; col++) {
        var a = get_pixel_channel(col - 1, row - 1, channel);
        var b = get_pixel_channel(col, row - 1, channel);
        var c = get_pixel_channel(col + 1, row - 1, channel);
        var d = get_pixel_channel(col - 1, row, channel);
        var e = get_pixel_channel(col, row, channel);
        var f = get_pixel_channel(col + 1, row, channel);
        var g = get_pixel_channel(col - 1, row + 1, channel);
        var h = get_pixel_channel(col, row + 1, channel);
        var i = get_pixel_channel(col + 1, row + 1, channel);

        set_pixel_channel(
          col, row, (a + b + c + d + f + g + h + i) % 256, channel
        );
      }
    }
  }

  updatePixels();
}
