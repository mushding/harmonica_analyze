(this["webpackJsonpfront-end"]=this["webpackJsonpfront-end"]||[]).push([[0],{46:function(e,t,a){e.exports=a(65)},51:function(e,t,a){},52:function(e,t,a){},57:function(e,t,a){},58:function(e,t,a){},59:function(e,t,a){},60:function(e,t,a){},61:function(e,t,a){},62:function(e,t,a){},65:function(e,t,a){"use strict";a.r(t);var n=a(0),r=a.n(n),o=a(21),i=a.n(o),c=a(3),l=a(4),s=a(6),u=a(5),m=a(7),d=a(42),h=a(16),f=(a(51),a(52),a(85)),p=a(35),v=a.n(p),g=function(e){function t(e){var a;return Object(c.a)(this,t),(a=Object(s.a)(this,Object(u.a)(t).call(this,e))).state={hidden:"hidden",checked:!1},a}return Object(m.a)(t,e),Object(l.a)(t,[{key:"componentWillMount",value:function(){var e=this;setTimeout((function(){e.show()}),e.props.wait)}},{key:"show",value:function(){var e,t=this;new Promise((function(t,a){setTimeout(t,e)})).then((function(){t.setState({hidden:""})})).then((function(){t.setState({checked:!0})}))}},{key:"render",value:function(){return r.a.createElement("div",{className:this.state.hidden},r.a.createElement(f.a,{in:this.state.checked,timeout:1800},r.a.createElement("div",null,r.a.createElement("h1",null,"Harmonica Analyzer"),r.a.createElement("p",null,"\u4e00\u500b\u7d66\u53e3\u7434\u521d\u5b78\u8005\u81ea\u6211\u6aa2\u9a57\u597d\u58de\u7684\u8a13\u7df4\u6a21\u578b",r.a.createElement("br",null),"\u53ef\u4ee5\u81ea\u52d5\u5206\u985e\u5439\u932f\u97f3\u3001\u97f3\u4e0d\u98fd\u6eff\u2026\u7b49\u554f\u984c"),r.a.createElement("a",{href:"https://github.com/mushding/harmonica_analyze",rel:"noopener noreferrer",target:"_blank"},r.a.createElement(v.a,null)))))}}]),t}(r.a.Component),b=function(e){function t(){return Object(c.a)(this,t),Object(s.a)(this,Object(u.a)(t).apply(this,arguments))}return Object(m.a)(t,e),Object(l.a)(t,[{key:"render",value:function(){return r.a.createElement("footer",{id:"footer"},r.a.createElement("p",{className:"copyright"},"\xa9 Untitled. Design: ",r.a.createElement("a",{href:"https://html5up.net"},"HTML5 UP"),"."))}}]),t}(r.a.Component),E=function(e){function t(){return Object(c.a)(this,t),Object(s.a)(this,Object(u.a)(t).apply(this,arguments))}return Object(m.a)(t,e),Object(l.a)(t,[{key:"componentDidMount",value:function(){}},{key:"render",value:function(){return r.a.createElement("div",null,r.a.createElement("div",{id:"wrapper"},r.a.createElement("header",{id:"header"},r.a.createElement("div",{className:"logo"},r.a.createElement("span",{className:"icon fa-chart-bar"})),r.a.createElement("div",{className:"content"},r.a.createElement("div",{className:"inner"},r.a.createElement(g,{wait:1e3}))),r.a.createElement("nav",null,r.a.createElement("ul",null,r.a.createElement("li",null,r.a.createElement("a",{href:"/#/start"},"start"))))),r.a.createElement(b,null)))}}]),t}(n.Component),w=(a(57),a(14)),S=(a(58),a(19)),C=a(79),y=Object(S.a)({root:{color:"#005266"}})(C.a),O=a(36),k=a.n(O),j=a(37),R=a.n(j),I=a(38),N=a.n(I),M=a(39),P=a.n(M),T=a(40),D=a.n(T),U=(a(59),a(60),function(e){function t(){return Object(c.a)(this,t),Object(s.a)(this,Object(u.a)(t).apply(this,arguments))}return Object(m.a)(t,e),Object(l.a)(t,[{key:"componentDidMount",value:function(){}},{key:"render",value:function(){var e=this.props.region;return r.a.createElement("div",{className:"messageContainer"},r.a.createElement("div",{className:"textContainer"},"1"===e.attributes?r.a.createElement("h3",null,"The Error is have a flat note"):"2"===e.attributes?r.a.createElement("h3",null,"The Error is have a double note"):r.a.createElement("h3",null,"attributes: ",e.attributes)),r.a.createElement("div",{className:"textContainer"},r.a.createElement("h3",null,"start at ",e.start," sec")),r.a.createElement("div",{className:"textContainer"},r.a.createElement("h3",null,"end at ",e.end," sec")))}}]),t}(r.a.Component)),x=a(80),A=a(81),z=a(82),W=a(83),B=a(84),L=function(e){function t(e){var a;return Object(c.a)(this,t),(a=Object(s.a)(this,Object(u.a)(t).call(this,e))).handdleRegionClick=function(e){console.log(e);var t=Object(w.a)(a),n=function(e){return new Promise((function(t,a){setTimeout(t,e)}))};n().then((function(){return t.setState({buttonRegionClick:!1}),n(1e3)})).then((function(){t.setState({regionContent:e})})).then((function(){t.setState({buttonRegionClick:!0})}))},a.initWavesurfer=function(){var e="https://www.haranalyzer.site/wav/"+String(a.props.file);a.createWavesurfer(),0===a.state.regionArr.length?a.createRegions():a.reCreateRegions(),a.props.isRecord?a.wavesurfer.load(a.props.blobUrl):a.wavesurfer.load(e)},a.createWavesurfer=function(){var e={barWidth:1,cursorWidth:1,container:"#waveform",backend:"MediaElement",height:200,progressColor:"#005266",responsive:!0,waveColor:"#fff",cursorColor:"#005266",plugins:[N.a.create()]},t=k.a.create(e);a.wavesurfer=t,a.createMinimap(),a.createTimeline(),a.createCursor(),t.on("region-in",(function(e){a.region=e})),t.on("region-out",a.onPlayEnd),t.on("region-click",a.handdleRegionClick)},a.onPlayEnd=function(){a.state.buttonIsLoop?a.wavesurfer.play():a.wavesurfer.play(a.region.start)},a.createRegions=function(){return fetch("https://www.haranalyzer.site/getoutput/"+a.props.file,{}).then((function(e){Promise.resolve(e.json()).then((function(e){return e})).then((function(e){a.setState({progressState:!1});for(var t=e.length,n=void 0,r=[],o=0;o<t;o++){var i=e[o];switch(e[o].type){case"1":n="hsla(197, 40%, 23%, 0.3)";break;case"2":n="hsla(195, 63%, 23%, 0.2)"}var c={start:i.start,end:i.end,drag:i.drag,resize:i.resize,attributes:i.type,color:n};r.push(c),a.wavesurfer.addRegion(c)}a.setState({regionArr:r})}))})).catch((function(e){return console.log(e),!1})),!0},a.reCreateRegions=function(){for(var e=a.state.regionArr.length,t=0;t<e;t++)a.wavesurfer.addRegion(a.state.regionArr[t])},a.reCreateWavesurfer=function(){a.wavesurfer.destroy(),a.initWavesurfer(),a.setState({buttonRegionClick:!1})},a.createMinimap=function(){a.wavesurfer.addPlugin(P.a.create({container:"#wave-minimap",waveColor:"#777",progressColor:"#222",height:50})).initPlugin("minimap")},a.createTimeline=function(){a.wavesurfer.addPlugin(D.a.create({container:"#wave-timeline"})).initPlugin("timeline")},a.createCursor=function(){a.wavesurfer.addPlugin(R.a.create({showTime:!0,opacity:1,hideOnBlur:!0,color:"#005266",customShowTimeStyle:{"background-color":"#005266",color:"#fff",padding:"3px","font-size":"15px"}})).initPlugin("cursor")},a.show=function(){var e=Object(w.a)(a),t=function(e){return new Promise((function(t,a){setTimeout(t,e)}))};t().then((function(){return e.setState({checked_1:!0}),t(800)})).then((function(){e.setState({checked_2:!0})}))},a.togglePlay=function(){var e=a.state.buttonIsPlay;a.wavesurfer.playPause(),a.setState({buttonIsPlay:!e})},a.toggleClearRegions=function(){var e=a.state.buttonIsShowRegions;e?a.wavesurfer.clearRegions():a.reCreateRegions(),a.setState({buttonIsShowRegions:!e})},a.toggleLoop=function(){var e=a.state.buttonIsLoop;a.setState({buttonIsLoop:!e})},a.toggleShowMiniMap=function(){var e=a.state.buttonIsShowMiniMap;e?a.wavesurfer.destroyPlugin("minimap"):a.createMinimap(),a.setState({buttonIsShowMiniMap:!e})},a.toggleShowTimeline=function(){var e=a.state.buttonIsShowTimeline;e?a.wavesurfer.destroyPlugin("timeline"):a.createTimeline(),a.setState({buttonIsShowTimeline:!e})},a.toggleShowCursor=function(){var e=a.state.buttonIsShowCursor;e?a.wavesurfer.destroyPlugin("cursor"):a.createCursor(),a.setState({buttonIsShowCursor:!e})},a.state={regionArr:[],regionContent:{},buttonIsPlay:!1,buttonIsLoop:!1,buttonIsShowRegions:!0,buttonIsShowMiniMap:!0,buttonIsShowTimeline:!0,buttonIsShowCursor:!0,buttonRegionClick:!1,checked_1:!1,checked_2:!1,progressState:!0},a}return Object(m.a)(t,e),Object(l.a)(t,[{key:"componentDidMount",value:function(){this.initWavesurfer();var e=this;setTimeout((function(){e.show()}),e.props.wait)}},{key:"render",value:function(){return r.a.createElement("div",null,r.a.createElement(f.a,{in:this.state.checked_1,timeout:1300},r.a.createElement("div",{className:"waveCursorContainer"},r.a.createElement("div",{className:"waveContainer"},r.a.createElement("div",{id:"waveform"}),r.a.createElement("br",null),r.a.createElement("div",{id:"wave-minimap"}),r.a.createElement("div",{id:"wave-timeline"})))),r.a.createElement(f.a,{in:this.state.checked_2,timeout:2e3},r.a.createElement("div",{className:"buttonContainer"},r.a.createElement("button",{onClick:this.togglePlay,disabled:this.state.progressState},r.a.createElement("div",{className:"buttonIcon"},this.state.buttonIsPlay?r.a.createElement(x.a,null):r.a.createElement(A.a,null))),r.a.createElement("button",{onClick:this.toggleLoop,disabled:this.state.progressState},r.a.createElement("div",{className:"buttonIcon"},this.state.buttonIsLoop?r.a.createElement(z.a,null):r.a.createElement(W.a,null))),r.a.createElement("button",{onClick:this.toggleClearRegions,disabled:this.state.progressState},this.state.buttonIsShowRegions?"disable regions":"able regions"),r.a.createElement("button",{onClick:this.toggleShowMiniMap,disabled:this.state.progressState},this.state.buttonIsShowMiniMap?"close minimap":"open minimap"),r.a.createElement("button",{onClick:this.toggleShowTimeline,disabled:this.state.progressState},this.state.buttonIsShowTimeline?"close timeline":"open timeline"),r.a.createElement("button",{onClick:this.toggleShowCursor,disabled:this.state.progressState},this.state.buttonIsShowCursor?"close cursor":"open sursor"),r.a.createElement("button",{onClick:this.reCreateWavesurfer,disabled:this.state.progressState},r.a.createElement("div",{className:"buttonIcon"},r.a.createElement(B.a,null))),this.state.progressState?r.a.createElement("div",{className:"progressContainer"},r.a.createElement(y,null)):r.a.createElement(f.a,{in:this.state.progressState,timeout:2e3},r.a.createElement("div",null)))),r.a.createElement(f.a,{in:this.state.buttonRegionClick,timeout:1e3},r.a.createElement(U,{region:this.state.regionContent})))}}]),t}(r.a.Component),_=function(e){function t(e){var a;return Object(c.a)(this,t),(a=Object(s.a)(this,Object(u.a)(t).call(this,e))).show=function(){var e=Object(w.a)(a),t=function(e){return new Promise((function(t,a){setTimeout(t,e)}))};t().then((function(){t(500),e.setState({checked:!0})}))},a.handleUploadImage=function(e){e.preventDefault();var t=new FormData;if(!a.uploadInput.files.length)return alert("Please select a wav file first."),!1;if(a.uploadInput.files[0].size>10485760)return alert("file size is more then 10 MBs."),!1;t.append("file",a.uploadInput.files[0]),a.setState({filename:a.uploadInput.files[0].name}),"idle"===a.state.progressState?(a.setState({progressState:"progress"}),fetch("https://www.haranalyzer.site/upload",{method:"POST",body:t}).then((function(e){if(e.ok)return a.setState({progressState:"success"}),e;throw new Error("Something went wrong")})).then((function(){console.log("IN!!!!"),a.setState({isUploadFile:!0})})).catch((function(e){console.log(e)}))):a.setState({progressState:"idle"})},a.toggleSelect=function(e){e.persist(),a.setState({filename:e.target.value})},a.toggleSelectButton=function(){if("default"===a.state.filename)return alert("Please select a example."),!1;a.setState({isUploadFile:!0})},a.state={checked:!1,isUploadFile:!1,filename:"default",progressState:"idle"},a}return Object(m.a)(t,e),Object(l.a)(t,[{key:"componentDidMount",value:function(){var e=this;setTimeout((function(){e.show()}),e.props.wait)}},{key:"render",value:function(){var e=this;return this.state.isUploadFile?r.a.createElement("div",null,r.a.createElement(L,{file:this.state.filename,isRecord:!1})):r.a.createElement("div",null,r.a.createElement(f.a,{in:this.state.checked,timeout:2e3},r.a.createElement("div",{className:"uploadContainer"},r.a.createElement("form",{onSubmit:this.handleUploadImage,className:"formContainer"},r.a.createElement("div",{className:"fromPadding"},r.a.createElement("label",{htmlFor:"file-upload",className:"custom-file-upload"},"\u8acb\u9078\u64c7 wav \u6a94"),r.a.createElement("input",{ref:function(t){e.uploadInput=t},type:"file",id:"file-upload",accept:"audio/wav"})),r.a.createElement("div",{className:"fromPadding"},r.a.createElement("button",{disabled:"progress"===this.state.progressState},"\u4e0a\u50b3"))),r.a.createElement("div",{className:"formContainer"},r.a.createElement("div",{className:"field"},r.a.createElement("label",{htmlFor:"demo-category"},"\u6216\u9078\u64c7\u7bc4\u4f8b\u97f3\u6a94"),r.a.createElement("select",{value:this.state.filename,name:"demo-category",id:"demo-category",onChange:this.toggleSelect},r.a.createElement("option",{value:"default"},"-"),r.a.createElement("option",{value:"little_star.wav"},"\u5c0f\u661f\u661f"))),r.a.createElement("div",{className:"fromPadding"},r.a.createElement("button",{onClick:this.toggleSelectButton,className:"selectButton",disabled:"progress"===this.state.progressState},"\u4e0a\u50b3"))),r.a.createElement("div",{className:"recordContainer"},r.a.createElement("div",{className:"recordTextContainer"},r.a.createElement("p",null,"\u6216\u9078\u64c7\u76f4\u63a5\u9304\u97f3")),r.a.createElement("div",{className:"uploadIcon"},r.a.createElement("a",{href:"/#/record"},r.a.createElement("button",null,"\u9032\u5165\u9304\u97f3\u9801\u9762")))),"success"===this.state.progressState?r.a.createElement("div",{className:"hidden"}):r.a.createElement(f.a,{in:"progress"===this.state.progressState,timeout:2e3,unmountOnExit:!0},r.a.createElement("div",{className:"progressContainer"},r.a.createElement(y,null))))))}}]),t}(r.a.Component),F=a(41),H=a.n(F),J=(a(61),function(e){function t(){return Object(c.a)(this,t),Object(s.a)(this,Object(u.a)(t).apply(this,arguments))}return Object(m.a)(t,e),Object(l.a)(t,[{key:"render",value:function(){return r.a.createElement("div",{className:"homeIcon"},r.a.createElement("a",{href:"/#/home"},r.a.createElement(H.a,null)))}}]),t}(r.a.Component)),G=function(e){function t(){return Object(c.a)(this,t),Object(s.a)(this,Object(u.a)(t).apply(this,arguments))}return Object(m.a)(t,e),Object(l.a)(t,[{key:"render",value:function(){return r.a.createElement("div",null,r.a.createElement(J,null),r.a.createElement(_,null))}}]),t}(r.a.Component),q=(a(62),null),K=null,Q=window.AudioContext||window.webkitAudioContext,V=null,X=null,Y=function(e){function t(e){var a;return Object(c.a)(this,t),(a=Object(s.a)(this,Object(u.a)(t).call(this,e))).getMicDevice=function(){navigator.mediaDevices.enumerateDevices().then((function(e){for(var t=[],n=0;n!==e.length;++n){var r=e[n];if("audioinput"===r.kind){a.setState({isMicAvailable:!0}),t.push(r);var o=r.label||"Microphone "+t.length;console.log("Mic ",o+" "+r.deviceId);var i=document.createElement("option");i.value=r.deviceId,i.text=o,a.micDeviceRef.current.appendChild(i)}}}))},a.drawSineWave=function(){var e=a.canvasRef.current.getContext("2d"),t=a.canvasRef.current.width,n=a.canvasRef.current.height,r=(a.canvasRef.current.width,a.canvasRef.current.height/2);X.fftSize=2048;var o=X.fftSize;console.log(o);var i=new Uint8Array(o);console.log(i),e.clearRect(0,0,t,n);!function(){X.getByteTimeDomainData(i),e.fillStyle="rgb(200, 200, 200)",e.fillRect(0,0,t,n),e.lineWidth=2,e.strokeStyle="rgb(0, 0, 0)",e.beginPath();for(var a=1*t/o,c=0,l=0;l<o;l++){console.log(i);var s=i[l]/128*n/2;0===l?e.moveTo(c,s):e.lineTo(c,s),c+=a}e.lineTo(t,r),e.stroke()}()},a.show=function(){var e=Object(w.a)(a),t=function(e){return new Promise((function(t,a){setTimeout(t,e)}))};t().then((function(){t(500),e.setState({checked:!0})}))},a.toggleStart=function(){if(a.state.isMicAvailable){var e={audio:{deviceId:{exact:a.micDeviceRef.current.value}},video:!1};navigator.mediaDevices.getUserMedia(e).then((function(e){(q=new MediaRecorder(e,{audioBitsPerSecond:705600,mimeType:"audio/webm"})).ondataavailable=a.handleDataAvailable,q.start(),a.setUpRecording(e)}))}else alert("There are no mic device.")},a.setUpRecording=function(e){V=new Q,V.sampleRate,V.createGain(),K=V.createMediaStreamSource(e),X=V.createAnalyser(),K.connect(X);var t=V.createScriptProcessor(2048,2,2);X.connect(t),t.connect(V.destination),a.drawSineWave()},a.handleDataAvailable=function(e){var t=e.data,n=new Blob([t],{type:"audio/wav"}),r=URL.createObjectURL(n);a.setState({blobUrl:r}),console.log(n),console.log(r);var o=new FormData;o.append("data",n),fetch("https://www.haranalyzer.site/recordUpload",{method:"POST",body:o}).then((function(e){a.setState({isStop:!0})}))},a.toggleStop=function(){q.stop()},a.state={isStop:!1,checked:!1,blobUrl:"",isMicAvailable:!1},a.micDeviceRef=r.a.createRef(),a.canvasRef=r.a.createRef(),a}return Object(m.a)(t,e),Object(l.a)(t,[{key:"componentDidMount",value:function(){this.getMicDevice();var e=this;setTimeout((function(){e.show()}),e.props.wait)}},{key:"render",value:function(){return this.state.isStop?r.a.createElement(L,{isRecord:!0,blobUrl:this.state.blobUrl,file:"record.wav"}):r.a.createElement("div",null,r.a.createElement(f.a,{in:this.state.checked,timeout:2e3},r.a.createElement("div",{className:"recordContainer"},r.a.createElement("select",{className:"selectContainer",ref:this.micDeviceRef,name:"",id:"micSelect"}),r.a.createElement("button",{className:"recordButtonContainer",onClick:this.toggleStart},"start record"),r.a.createElement("button",{className:"recordButtonContainer",onClick:this.toggleStop},"stop record")),r.a.createElement("canvas",{width:"500",height:"300",ref:this.canvasRef})))}}]),t}(r.a.Component),Z=function(e){function t(){return Object(c.a)(this,t),Object(s.a)(this,Object(u.a)(t).apply(this,arguments))}return Object(m.a)(t,e),Object(l.a)(t,[{key:"componentDidMount",value:function(){}},{key:"render",value:function(){return r.a.createElement("div",null,r.a.createElement(J,null),r.a.createElement(Y,null))}}]),t}(r.a.Component),$=function(e){function t(){return Object(c.a)(this,t),Object(s.a)(this,Object(u.a)(t).apply(this,arguments))}return Object(m.a)(t,e),Object(l.a)(t,[{key:"render",value:function(){return r.a.createElement(d.a,null,r.a.createElement(h.d,null,r.a.createElement(h.b,{exact:!0,path:"/",render:function(){return r.a.createElement(h.a,{to:"/home"})}}),r.a.createElement(h.b,{path:"/home",component:E}),r.a.createElement(h.b,{path:"/start",component:G}),r.a.createElement(h.b,{path:"/record",component:Z})))}}]),t}(n.Component);i.a.render(r.a.createElement($,null),document.getElementById("root"))}},[[46,1,2]]]);
//# sourceMappingURL=main.83cdd5bc.chunk.js.map