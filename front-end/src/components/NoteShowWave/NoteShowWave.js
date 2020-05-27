import React from 'react';
import PropTypes from 'prop-types';
import WaveSurfer from 'wavesurfer.js';
import CursorPlugin from 'wavesurfer.js/dist/plugin/wavesurfer.cursor';
import RegionsPlugin from 'wavesurfer.js/dist/plugin/wavesurfer.regions';
import MinimapPlugin from 'wavesurfer.js/dist/plugin/wavesurfer.minimap';
import TimelinePlugin from 'wavesurfer.js/dist/plugin/wavesurfer.timeline';
import './styles.css';
import RegionMessage from '../RegionMessage/RegionMessage'
import Collapse from '@material-ui/core/Collapse';
import ColorCircularProgress from '../ProgressCircle/ProgressCircle'
import {
    PlayArrow,
    Pause,
    Refresh,
} from '@material-ui/icons'

export default class NoteShowWave extends React.Component{
    constructor(props){
        super(props);
        this.state = {
            userRegionArr: [],
            correctRegionArr: [],
            regionContent: {},
            buttonIsPlay: false,
            buttonIsShowUserRegions: true,
            buttonIsShowCorrectRegions: true,
            buttonIsShowMiniMap: true,
            buttonIsShowTimeline: true,
            buttonRegionClick: false,
            checked_1: false,
            checked_2: false,
            progressState: true,
        }
    }
    handdleRegionClick = (region) => {
        console.log(region)
        let that = this
		let delay = (s) => {
            return new Promise((resolve, reject) => {
                setTimeout(resolve, s); 
            });
        };
        delay().then(() => {
            that.setState({ buttonRegionClick: false })
            return delay(1000); 
        }).then(() => {
            that.setState({ regionContent: region })
        }).then(() => {
            that.setState({ buttonRegionClick: true })
        })
    }
    initWavesurfer = () => {
        const url = "https://www.haranalyzer.site/wav/" + String(this.props.file)
        this.wavesurferArray = []
        // wavesurferArray[0] -> user, wavesurferArray[1] -> correct
        this.wavesurferArray.push(this.createWavesurfer('#user'))
        this.wavesurferArray.push(this.createWavesurfer('#correct'))
        if (this.state.userRegionArr.length === 0) {
            this.createRegions("getCorrectRegions")
            this.createRegions("getUserRegions")
        } else {
            this.reCreateUserRegions()
            this.reCreateCorrectRegions()
        }
        this.wavesurferArray[0].load(url)
        this.wavesurferArray[1].load(url)
    }
    createWavesurfer = (waveformType) => {
        const options = {
            barWidth: 1,
            cursorWidth: 1,
            container: waveformType + "-waveform",
            backend: 'MediaElement',
            height: 200,
            progressColor: '#005266',
            responsive: true,
            waveColor: '#fff',
            cursorColor: '#005266',
            plugins: [
                RegionsPlugin.create(),
            ]
        };
        const wavesurfer = WaveSurfer.create(options);
        this.createTimeline(wavesurfer, waveformType)
        this.createCursor(wavesurfer)

        // event listener
        wavesurfer.on('region-in', (region) => {
            this.region = region
        })
        wavesurfer.on('region-click', this.handdleRegionClick)
        return wavesurfer
    }
    createRegions = (regionType) => {
        const NOTE_COLOR_TABLE = {
            'C5': "hsla(180, 100%, 77%, 0.45)",
            'D5': "hsla(229, 100%, 77%, 0.45)",
            'E5': "hsla(117, 100%, 77%, 0.45",
            'F5': "hsla(139, 100%, 50%, 0.45)",
            'G5': "hsla(62, 100%, 77%, 0.45)",
            'A5': "hsla(35, 100%, 77%, 0.45)",
            'B5': "hsla(5, 100%, 77%, 0.45)",
            'C6': "hsla(311, 100%, 77%, 0.45)",
        }
        fetch("https://www.haranalyzer.site/" + regionType + "/" + this.props.file + "/" + this.props.mxlfile, {})
            .then((response) => {
                let data_promise = Promise.resolve(response.json())
                data_promise.then((data) => {
                    return data
                }).then((jsonData) => {
                    this.setState({ progressState: false })
                    let len = jsonData.length
                    let regionStateArr = []
                    for (let i = 0; i < len; i++){
                        const regionData = jsonData[i]
                        const errorType = jsonData[i]["type"]
                        const regionColor = NOTE_COLOR_TABLE[errorType]
                        const regionDir = {
                            start: regionData["start"],
                            end: regionData["end"],
                            drag: regionData["drag"],
                            resize: regionData["resize"],
                            attributes: regionData["type"],
                            color: regionColor,
                        }
                        regionStateArr.push(regionDir)
                        if (regionType === "getUserRegions"){
                            this.wavesurferArray[0].addRegion(regionDir)
                        } else {
                            this.wavesurferArray[1].addRegion(regionDir)
                        }
                    }
                    if (regionType === "getUserRegions"){
                        this.setState({ userRegionArr: regionStateArr})
                    } else {
                        this.setState({ correctRegionArr: regionStateArr})
                    }
                })
            })
            .catch((error) => {
                console.log(error)
                return false
            })
        return true
    }
    reCreateUserRegions = () => {
        let arrLen = this.state.userRegionArr.length
        for(let i = 0; i < arrLen; i++){
            this.wavesurferArray[0].addRegion(this.state.userRegionArr[i]) 
        }
    }
    reCreateCorrectRegions = () => {
        let arrLen = this.state.correctRegionArr.length
        for(let i = 0; i < arrLen; i++){
            this.wavesurferArray[1].addRegion(this.state.correctRegionArr[i]) 
        }
    }
    reCreateWavesurfer = () => {
        this.wavesurferArray[0].destroy()
        this.wavesurferArray[1].destroy()
        this.initWavesurfer()
        this.setState({ buttonRegionClick: false })
    }
    createTimeline = (wavesurfer, waveformType) => {
        wavesurfer.addPlugin(TimelinePlugin.create({
            container: waveformType + "-wave-timeline",
        })).initPlugin('timeline')
    }
    createCursor = (wavesurfer) => {
        wavesurfer.addPlugin(CursorPlugin.create({
            showTime: true,
            opacity: 1,
            hideOnBlur: true,
            color: "#005266",
            customShowTimeStyle: {
                'background-color': '#005266',
                color: '#fff',
                padding: '3px',
                'font-size': '15px'
            }
        })).initPlugin('cursor')
    }
    componentDidMount(){
        this.initWavesurfer()
        var that = this;
		setTimeout(() => {
			that.show();
		}, that.props.wait);
    }
    show = () => {
        let that = this
		let delay = (s) => {
            return new Promise((resolve, reject) => {
                setTimeout(resolve, s); 
            });
        };
        delay().then(() => {
            that.setState({checked_1: true})
            return delay(800); 
        }).then(() => {
            that.setState({checked_2: true})
        });
    }
    togglePlay = () => {
        const bool = this.state.buttonIsPlay
        this.wavesurferArray[1].playPause();
        this.setState({ buttonIsPlay: !bool })
    }
    toggleClearUserRegions = () => {
        const bool = this.state.buttonIsShowUserRegions
        bool ? this.wavesurferArray[0].clearRegions() : this.reCreateUserRegions()
        this.setState({ buttonIsShowUserRegions: !bool })
    }
    toggleClearCorrectRegions = () => {
        const bool = this.state.buttonIsShowCorrectRegions
        bool ? this.wavesurferArray[1].clearRegions() : this.reCreateCorrectRegions()
        this.setState({ buttonIsShowCorrectRegions: !bool })
    }
    toggleShowTimeline = () => {
        const bool = this.state.buttonIsShowTimeline
        if (bool){
            this.wavesurferArray[0].destroyPlugin('timeline') 
            this.wavesurferArray[1].destroyPlugin('timeline')  
        } else {
            this.createTimeline(this.wavesurferArray[0], "#user")
            this.createTimeline(this.wavesurferArray[1], "#correct")
        }
        this.setState({ buttonIsShowTimeline: !bool})
    }
    render(){
        return (
            <div>
                <Collapse in={this.state.checked_1} timeout={2000}> 
                    <h3 className="textContainer">剛剛錄音所判斷出的音高及長度</h3>
                    <div className="waveCursorContainer">
                        <div className="waveContainer">
                            <div id='user-waveform'></div>
                            <br/>
                            <div id='user-wave-timeline'></div>
                        </div>
                    </div>
                    <h3 className="textContainer">mxl 檔的正確音高及長度</h3>
                    <div className="waveCursorContainer">
                        <div className="waveContainer">
                            <div id='correct-waveform'></div>
                            <br/>
                            <div id='correct-wave-timeline'></div>
                        </div>
                    </div>
                </Collapse>
                <Collapse in={this.state.checked_2} timeout={2000}>
                    <div className="buttonContainer">
                        <button onClick={this.togglePlay} disabled={this.state.progressState}>
                            <div className="buttonIcon">
                                { this.state.buttonIsPlay ? <Pause/> : <PlayArrow/> }
                            </div>
                        </button>
                        <button onClick={this.toggleClearUserRegions} disabled={this.state.progressState}>
                            { this.state.buttonIsShowUserRegions ? "關閉上面波形圖的標籤" : "開起上面波形圖的標籤" }
                        </button>
                        <button onClick={this.toggleClearCorrectRegions} disabled={this.state.progressState}>
                            { this.state.buttonIsShowCorrectRegions ? "關閉下面波形圖的標籤" : "開起下面波形圖的標籤" }
                        </button>
                        <button onClick={this.toggleShowTimeline} disabled={this.state.progressState}>
                            { this.state.buttonIsShowTimeline ? "關閉時間軸" : "開啟時間軸" }
                        </button>
                        <button onClick={this.reCreateWavesurfer} disabled={this.state.progressState}>
                            <div className="buttonIcon">
                                <Refresh/>
                            </div>
                        </button>
                        { this.state.progressState  ? (
                            <div className="progressContainer">
                                <ColorCircularProgress/>
                            </div>
                        ) : (
                            <Collapse in={this.state.progressState} timeout={2000}>
                                <div></div>
                            </Collapse>
                        ) }
                    </div>
                </Collapse>
                <Collapse in={this.state.buttonRegionClick} timeout={1000}>
                    <RegionMessage region={this.state.regionContent}/>
                </Collapse>    
            </div>
        );
    }
}

NoteShowWave.propTypes = {
    file: PropTypes.string,
    blobUrl: PropTypes.string
}