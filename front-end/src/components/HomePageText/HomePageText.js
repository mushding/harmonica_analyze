import React from 'react'
import './styles.css'
import Collapse from '@material-ui/core/Collapse';
import GitHubIcon from '@material-ui/icons/GitHub';

export default class HomePageText extends React.Component{
	constructor(props){
        super(props)
        this.state = {
            hidden: "hidden",
            checked: false,
        }
    }
	componentWillMount(){
        var that = this;
		setTimeout(function() {
			that.show();
		}, that.props.wait);
	}
	show() {
        var that = this
		var delay = function(s){
            return new Promise(function(resolve, reject){
                setTimeout(resolve,s); 
            });
        };
        delay().then(function(){
            that.setState({hidden: ""})
            // return delay(200); 
        }).then(function(){
            that.setState({checked: true})
        });
	}
    render() {
        return (
            <div className={this.state.hidden}>
                <Collapse in={this.state.checked} timeout={1800}>
                    <div>
                        <h1>Harmonica Analyzer</h1>
                        <p>一個給口琴初學者自我檢驗好壞的訓練模型<br/>
                        可以自動分類吹錯音、音不飽滿…等問題</p>
                        <a href="https://github.com/mushding/harmonica_analyze" rel="noopener noreferrer" target="_blank"><GitHubIcon/></a>
                    </div>
                </Collapse>
            </div>
        )
    }
};