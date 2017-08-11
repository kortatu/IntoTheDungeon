$('document').ready(function() {
    window.setInterval(function(){
        classify(capture()).then(function(response){
        document.getElementById('topCategory').innerHTML = response.topCategory;
        document.getElementById('score').innerHTML = response.score;

        console.log(response);
        });
    }, 500);
});


function capture() {
    var canvas = document.getElementById('canvas');
    var video = document.getElementById('video');
    canvas.getContext('2d').drawImage(video, 0, 0);
    return canvas.toDataURL('image/jpeg', 1.0);
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