$('document').ready(function() {
    window.setInterval(function(){
        classify(capture()).then(function(response){
            console.log(response);
        });
    }, 250);
});


function capture() {
    var canvas = document.getElementById('canvas');
    var video = document.getElementById('video');
    canvas.getContext('2d').drawImage(video, 0, 0);
    return canvas.toDataURL();
}

function classify( thumbnail ) {
    return $.ajax({
        type: "POST",
        url: '/api/v1/classify',
        data: JSON.stringify({ imgBase64: thumbnail.replace(/^data:image\/[a-z]+;base64,/, "") }),
        dataType: 'json',
        contentType: "application/json"
    });
}