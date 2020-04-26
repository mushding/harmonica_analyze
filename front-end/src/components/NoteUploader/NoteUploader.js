import React from 'react'
import Collapse from '@material-ui/core/Collapse';
import ColorCircularProgress from '../ProgressCircle/ProgressCircle'
import NoteRecorder from '../NoteRecorder/NoteRecorder'
import './styles.css'


export default class NoteUploader extends React.Component{
    constructor(props){
        super(props)
        this.state = {
            isUploadFile: false,
            checked: false,
            progressState: 'idle',
            filename: "default",
        }
    }
    componentDidMount() {
        var that = this;
		setTimeout(() => {
			that.show();
		}, that.props.wait);
    }
    show = () => {
        var that = this
		var delay = (s) => {
            return new Promise((resolve, reject) => {
                setTimeout(resolve,s); 
            });
        };
        delay().then(() => {
            delay(500)
            that.setState({ checked: true })
        })
    }
    handleUploadImage = (ev) => {
        // handle read data from form
        ev.preventDefault();
        const data = new FormData();
        
        // handle file max size
        if (this.uploadInput.files.length){
            const maxSize = 1024 * 1024 * 10
            let fileSize = this.uploadInput.files[0].size
            if (fileSize > maxSize){
                alert("file size is more then " + 10 + " MBs.")
                return false
            }
        } else {
            alert('Please select a mxl file first.')
            return false
        }
        data.append('file', this.uploadInput.files[0]);
        this.setState({ filename: this.uploadInput.files[0].name })
        
        // handle progress circle
        if (this.state.progressState !== 'idle') {
            this.setState({ progressState: 'idle' });
            return;
        }
        this.setState({ progressState: 'progress' });
        

        fetch('https://www.haranalyzer.site/uploadmxl', {
            method: 'POST',
            body: data,
        }).then((response) => {
            if(response.ok){
                this.setState({ progressState: 'success' })
                return (response)
            } else {
                throw new Error('Something went wrong');
            }
        }).then(() => {
            console.log("IN!!!!")
            this.setState({ isUploadFile: true })
        }).catch((error) => {
            console.log(error)
        })
    }
    render(){
        if (!this.state.isUploadFile){
            return(
                <div>
                    <Collapse in={this.state.checked} timeout={2000}>
                        <div className="recordContainer">
                            <p>請先上傳 mxl 檔案到網站上，上傳完畢後會自動跳轉到錄音頁面</p>
                        </div>
                        <div className="recordContainer">
                            <div className="recordTextContainer">
                                <p>不知道什麼是 mxl 檔案嗎？</p>
                            </div>
                            <a href="https://hackmd.io/neoAtjS4RQegjdL_r3ucug" rel="noopener noreferrer" target="_blank"><button>按這進入如何製作 mxl 檔案教學</button></a>
                        </div>
                        <div className="uploadContainer">
                            <form onSubmit={this.handleUploadImage} className="formContainer">
                                <div className="fromPadding">
                                    <label htmlFor="file-upload" className="custom-file-upload">
                                        請選擇 mxl 檔
                                    </label>
                                    <input ref={(ref) => { this.uploadInput = ref; }} type="file" id="file-upload" accept=".mxl"/>
                                </div>
                                <div className="fromPadding">
                                    <button disabled={this.state.progressState === 'progress'}>上傳</button>
                                </div>
                            </form>
                            {this.state.progressState === 'success' ? (
                                <div className="hidden"></div>
                            ) : (
                                <Collapse
                                    in={this.state.progressState === 'progress'}
                                    timeout={2000}
                                    unmountOnExit
                                >
                                    <div className="progressContainer">
                                        <ColorCircularProgress />
                                    </div>
                                </Collapse>
                            )}
                        </div>
                    </Collapse>
                </div>     
            );
        } else {
            return(
                <NoteRecorder mxlfile={this.state.filename}/>
            );
        }
    }
}